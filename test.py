from roboflow import Roboflow
rf = Roboflow(api_key="xryMW9j32H1tQ20ehfWF")
project = rf.workspace().project("american-sign-language-letters-l980l")
model = project.version(1).model

# infer on a local image
print(model.predict("arduino_hc_06_connection.jpg", confidence=40, overlap=30).json())