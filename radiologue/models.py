from django.db import models

class Radiologue(models.Model):
    nom = models.CharField(max_length=100)
    specialite = models.CharField(max_length=100)
    image = models.FileField(upload_to='static/media/luna', default='default_image.jpg')  # Utiliser FileField pour gérer les fichiers .npy


    def __str__(self):
        return self.nom
