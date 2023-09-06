import dash
from dash import dcc as dcc
from dash import html as html
from dash.dependencies import Input, Output
from detect_recog import extract_number
import base64
import io
from PIL import Image
import os

# Import the functions from the script
from detect_recog import ocr
from encoding import image_to_base64

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(style={
    'textAlign': 'center',
    'margin': 'auto',
    'width': '70%',
}, children=[
    html.H1('OCR(YOLOv5+TRBA)'),
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Images')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=True  # This allows multiple file uploads
    ),
    dcc.Loading(  # Adding the Loading component
        id="loading-div",
        children=[html.Div(id='output-image-upload')],
        type="dot"  # or "cube", "circle", "dot", depending on your preference
    )
])

@app.callback(
    Output('output-image-upload', 'children'),
    [Input('upload-image', 'contents')]
)
def update_output(contents):
    children = []
    dt_model = 'iou050_size960.pt'
    re_model = 'iter_50000.pth'

    if contents is not None:
        # OCR이 실행되었는지 확인하는 부울 변수
        ocr_executed = False

        for idx, content in enumerate(contents):
            # 이미지 내용을 파싱합니다.
            content_type, content_string = contents[idx].split(',')
            decoded = base64.b64decode(content_string)
            image = Image.open(io.BytesIO(decoded))

            # 고유한 이름으로 임시 위치에 이미지를 저장합니다.            
            image_path = "image/"
            if not os.path.exists(image_path):
                os.makedirs(image_path)
            image_name = f"image_{idx}.jpg"
            image.save(image_path+image_name)
            
            # children.append(html.Div([
            #     html.Img(src=content, style={'width': '50%'}),  # 이미지 미리보기
            #     html.Br(),
            # ]))

        if not ocr_executed:
        # ocr 함수를 실행합니다.
            ocr(f'./yolov5/{dt_model}', image_path, re_model)
            
        # detection box 이미지 불러오기
        detect_path = 'yolov5/runs/detect'
        # detect_list
        detect_list = os.listdir(detect_path)
        sorted_detect_list = sorted(detect_list, key=extract_number)
        last_detect = sorted_detect_list[-1]
        detect_image_path = (f'{detect_path}/{last_detect}/')
        jpeg_files = [f for f in os.listdir(detect_image_path) if f.lower().endswith(('.jpg', '.jpeg'))]
        for jpeg_file in jpeg_files:
            encoded_image = image_to_base64(os.path.join(detect_image_path, jpeg_file))
            decoded = base64.b64decode(encoded_image)
            image = Image.open(io.BytesIO(decoded))
            children.append(html.Div([
                html.Img(src=image, style={'width': '50%'}),  # 이미지 미리보기
                html.Br(),
            ]))

        # demo_log.txt의 내용을 읽습니다.
        with open("deep-text-recognition-benchmark/demo_log.txt", "r", encoding='utf-8') as file:
            log_content = file.read()
            ocr_executed = True
        
        children.append(html.Div([f'detection model : ( iou : {int(dt_model[3:6])*0.01}, size : {dt_model[-6:-3]} )',
                                  html.Br(),
                                  f'recognition model : ( epoch : {re_model.split("_")[1].split(".")[0]} )']),)
        
        
        # 모든 이미지 후에 로그 내용을 한 번만 추가합니다.
        children.append(html.Div([
            # html.Br(),
            html.Pre(log_content),  # 공백과 줄바꿈을 유지하기 위해 <pre> 태그를 사용합니다.
            html.Hr()  # 구분을 위한 수평선
        ]))

    else:
        return html.Div('이미지를 업로드해주세요.')

    return children



if __name__ == '__main__':
    app.run_server(debug=True)
