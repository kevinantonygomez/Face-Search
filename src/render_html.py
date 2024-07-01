import os

class Renderer:
    '''
    Creates HTML file to display results.
    Renders the query image and all found similar images
    along with their paths and scores
    '''
    def __init__(self) -> None:
        self.html_template = '''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Image Grid</title>
            <style>
                body {{
                    background-color: black;
                    color: white;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    padding: 20px;
                }}
                .query-image-container {{
                    text-align: center;
                    margin-bottom: 20px;
                    width: 228px;  
                    max-width: 800px; 
                    height: auto;  
                }}
                .query-image-container img {{
                    max-width: 100%;
                    height: auto;
                }}
                .caption {{
                    margin-top: 5px;
                    font-size: 0.9em;
                    color: white;
                }}
                .grid-container {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 10px;
                    width: 100%;
                    max-width: 1200px;
                }}
                .grid-item {{
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                    border: 1px solid #ccc;
                    padding: 10px;
                    text-align: center;
                    background-color: black;
                    color: white;
                }}
                .grid-item img {{
                    max-width: 100%;
                    max-height: 100%;
                }}
            </style>
        </head>
        <body>
            <div class="query-image-container">
                <img src="{query_image}" alt="Query Image">
                <div class="caption" style="word-break: break-word">
                <a href= "{query_image}"> {query_image} </a>
                </div>
            </div>
            <div class="grid-container">
                {image_tags}
            </div>
        </body>
        </html>
        '''
        self.query_image = ''
        self.image_tags = ''
    
    def set_query_image(self, path) -> None:
        '''
        :param path: query image path
        '''
        if not self._type_check('path', path, str):
            raise TypeError 
        if not os.path.isfile(path):
            raise FileNotFoundError
        self.query_image = path

    def update_image_tags(self, path:str, score:float) -> None:
        '''
        Updates the image tag to display the passed image and score
        :param path: image path
        :param score: similarity score for this image
        '''
        if not self._type_check('path', path, str) or not self._type_check('score', score, float):
            raise TypeError 
        if not os.path.isfile(path):
            raise FileNotFoundError
        self.image_tags =  f'''
        {self.image_tags}
            <div class="grid-item">
                <img src="{path}" alt="Image">
                <div class="caption" style="word-break: break-word;">
                <a href= "{path}"> {path} </a>
                <br>
                <b> Score: {score} </b>
                </div>
            </div>'''

    def render(self, html_path="data/image_grid.html") -> None:
        '''
        Formats and outputs HTML file
        :param html_path: output HTML path. Creates one if doesn't exist
        :param score: similarity score for this image
        '''
        if not self._type_check('html_path', html_path, str):
            raise TypeError 
        html_content = self.html_template.format(query_image=self.query_image, image_tags=self.image_tags)
        try:
            with open(html_path, 'w+') as file:
                file.write(html_content)
            print('HTML file written')
        except Exception as e:
            print(f'render fail:\n{e}')
    
    def _type_check(self, obj_name:str, obj, type)->bool:
        '''
        Private method to type check an object
        :param obj_name: variable name
        :param obj: actual obj
        :param type: expected type of obj
        '''
        try:
            if not isinstance(obj, type):
                print(f'!!! {obj_name} should be of type {type} not {type(obj)}')
            else:
                return True
        except Exception as e:
            print(f"!!! Error in _type_check: \n {e}")
        return False