import os
from utils.logger_handler import logger
import hashlib
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader



def get_file_md5_hex(filepath: str):
    if not os.path.exists(filepath):
        logger.error(f"[md5计算] 文件{filepath}不存在")
        return
    if not os.path.isfile(filepath):
        logger.error(f"[md5计算] {filepath} 不是一个文件")
        return
    md5_obj = hashlib.md5()

    chunk_size = 4096
    try:
        with open(filepath, "rb") as f:
            while chunk := f.read(chunk_size):
                md5_obj.update(chunk)
            md5_hex = md5_obj.hexdigest()
            return md5_hex
    except Exception as e:
        logger.error(f"[md5计算] 文件{filepath}计算失败, {str(e)}")
        return None

def listdir_with_allowed_type(path: str, allowed_types: tuple[str]):  #传入的是一个路径
    files=[]

    if not os.path.isdir(path):
        logger.error(f"[文件列表] {path} 不是一个目录")
        return allowed_types
    
    for f in os.listdir(path):
        if f.endswith(allowed_types):
            files.append(os.path.join(path, f))

    return tuple(files)
def pdf_loader(filepath: str, passwd=None) -> list[Document]:
    return PyPDFLoader(filepath, passwd).load()

def txt_loader(filepath: str) -> list[Document]:
    return TextLoader(filepath, encoding="utf-8").load()