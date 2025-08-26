from rest_framework.views import APIView
from django.core.files.storage import FileSystemStorage
from rest_framework.response import Response
from .predictions import load_model_from_checkpoint


class ImageView(APIView):

    def post(self, request, *args, **kwargs):
        if 'image' in request.data:
            image = request.data['image']
            path_to_model = 'catdogmodel.pt'
            label = load_model_from_checkpoint(image, path_to_model)
            if str(label) == 'tensor(0)':
                label = 'cat'
            elif str(label) == 'tensor(1)':
                label = 'dog'
            fs = FileSystemStorage()
            saved_image = fs.save(image.name, image)

            image_url = f'http://127.0.0.1:8000/media/{saved_image}'
            return Response({'message':f'your predicted label is {label}','image':image_url},status=200)




    
