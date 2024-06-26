import json
import logging
from pathlib import Path
from typing import Iterator, Optional, Union

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.helpers import detect_file_encodings

logger = logging.getLogger(__name__)


class JD_DOC_Loader(BaseLoader):
    """Load text file.


    Args:
        file_path: Path to the file to load.

        encoding: File encoding to use. If `None`, the file will be loaded
        with the default system encoding.

        autodetect_encoding: Whether to try to autodetect the file encoding
            if the specified encoding fails.
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        encoding: Optional[str] = None,
        autodetect_encoding: bool = False,
    ):
        """Initialize with file path."""
        self.file_path = file_path
        self.encoding = encoding
        self.autodetect_encoding = autodetect_encoding

    def lazy_load(self) -> Iterator[Document]:
        """Load from file path."""
        text = ""
        from_url = ""
        try:
            with open(self.file_path, encoding=self.encoding) as f:
                doc_data = json.load(f)
                text = doc_data["content"]
                title = doc_data["title"]
                product = doc_data["product"]
                from_url = doc_data["url"]

                # text = f.read()
        except UnicodeDecodeError as e:
            if self.autodetect_encoding:
                detected_encodings = detect_file_encodings(self.file_path)
                for encoding in detected_encodings:
                    logger.debug(f"Trying encoding: {encoding.encoding}")
                    try:
                        with open(self.file_path, encoding=encoding.encoding) as f:
                            text = f.read()
                        break
                    except UnicodeDecodeError:
                        continue
            else:
                raise RuntimeError(f"Error loading {self.file_path}") from e
        except Exception as e:
            raise RuntimeError(f"Error loading {self.file_path}") from e
        # metadata = {"source": str(self.file_path)}
        metadata = {"source": from_url, "title": title, "product": product}
        yield Document(page_content=text, metadata=metadata)
