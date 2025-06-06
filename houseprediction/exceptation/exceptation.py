import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from houseprediction.logging import logger


class housepredException(Exception):
    """Custom exception class to provide detailed error messages."""
    def __init__(self, error_message, error_detail: sys):
        
    
        self.error_message= error_message
        _,_,exc_tb= error_detail.exc_info()
        self.lineno= exc_tb.tb_lineno
        self.file_name= exc_tb.tb_frame.f_code.co_filename
        
    def __str__(self):
        return "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
            self.file_name,self.lineno ,str(self.error_message)
        )
         
if __name__=='__main__':
    try:
        logger.logging.info("Enter the try block")
        a=1/0
        print("This will not printed",a)
    except Exception as e:
        raise housepredException(e,sys)