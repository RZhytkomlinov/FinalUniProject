from rest_framework.views import APIView
from django.core.files.storage import FileSystemStorage
#from rest_framework.parsers import MultiPartParser, FormParser
from django.http import JsonResponse
from django.core.files import File
#from .serializers import ImageSerializer
from rest_framework.response import Response
#from rest_framework import status
#import base64
# Create your views here.

class ImageView(APIView):
    #parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        if 'image' in request.data:
            image = request.data['image']
            name = request.data['name']
            fs = FileSystemStorage()
            saved_image = fs.save(image.name, image)
            print(saved_image)
            image_url = f'http://127.0.0.1:8000/media/{saved_image}'
            #print(type(image))
            #processed_image = base64.b64decode(File(image).read())
            #processed_image = ImageSerializer(image)
            #print(processed_image.is_valid())
            #if processed_image.is_valid():
                #print(processed_image)
                #print(processed_image)
            return Response({'message':'your predicted label gonna be here','image':image_url},status=200)
            print('lets fucking go')



'''      image_serializer = ImageSerializer(data=request.data)
        print(request)
        if image_serializer.is_valid():
            image_serializer.save()
            print('thank god yes')
            return Response(image_serializer.data, status=status.HTTP_201_CREATED)
        else:
            return Response(image_serializer.errors, status=status.HTTP_400_BAD_REQUEST)'''
        
    