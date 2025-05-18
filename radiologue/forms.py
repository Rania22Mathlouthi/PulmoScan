from django import forms
from .models import Radiologue

class UploadImageForm(forms.Form):
    image = forms.FileField(widget=forms.FileInput(), required=True)  # Simple file upload field

    def clean(self):
        cleaned_data = super().clean()
        image = cleaned_data.get('image')

        if image:
            # Validate file type (ensure it's .npy)
            if not image.name.endswith('.npy'):
                self.add_error('image', "Le fichier doit être au format .npy.")

        return cleaned_data
