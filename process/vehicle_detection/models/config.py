from pydantic import BaseModel
from

class ImagePaths(BaseModel):
    # main images
    init_img: str = gui_init_image_path
    login_img: str = login_button_image_path