import os
import shutil
import uuid
from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
    HTTPException,
    status,
)

from app.services import document_parser, vector_store

router = APIRouter()

# Define a directory to store temporary uploads
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...), 
    session_id: str = Form(...)
):
    """
    Handles file uploads for RAG.

    This endpoint:
    1.  Receives a file and a session_id.
    2.  Saves the file to a temporary location with a unique name.
    3.  Calls the document_parser service to extract content into chunks.
    4.  Calls the vector_store service to embed and store the chunks.
    5.  Cleans up the temporary file.
    6.  Returns a success or error response.
    """
    
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file name provided."
        )

    # Create a unique, secure filename to avoid collisions and path traversal
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)

    try:
        # Save the uploaded file to the temporary directory
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # --- Core Logic: Parse and Embed ---
        print(f"Parsing file: {file.filename} ({file_path})")
        document_chunks = document_parser.parse_file(file_path, file_extension)

        if not document_chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not parse any content from the file: {file.filename}"
            )

        print(f"Adding {len(document_chunks)} chunks to the vector store.")
        print(f"Adding chunks to vector store for session_id: {session_id}")
        vector_store.add_documents_to_store(document_chunks, session_id)
        # vector_store.add_documents_to_store(document_chunks)
        
        # Here you could also link the document IDs/chunks to the session_id in Redis/Postgres
        # For now, we are adding to a global store for simplicity.

        return {
            "status": "success",
            "filename": file.filename,
            "chunks_added": len(document_chunks),
            "message": "File processed and added to the knowledge base successfully."
        }

    except HTTPException as e:
        # Re-raise HTTP exceptions to be handled by FastAPI
        raise e
    except Exception as e:
        # Catch any other unexpected errors during processing
        print(f"Error processing file {file.filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred while processing the file: {e}"
        )
    finally:
        # --- Cleanup: Ensure the temporary file is always deleted ---
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Cleaned up temporary file: {file_path}")
        
        # Ensure the file stream is closed
        await file.close()