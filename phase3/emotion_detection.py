from zipfile import ZipFile

data = 'phase3/dataset/fer2013.zip'     
with ZipFile(data, 'r') as zip_ref:
    zip_ref.extractall('dataset/')   
    print("Unzipping completed!")
