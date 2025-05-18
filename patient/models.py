from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator, FileExtensionValidator, URLValidator
from django.utils.html import mark_safe
import datetime

class Patient(models.Model):
    GENRE_CHOICES = [
        ('Action', 'Action'),
        ('Comedy', 'Comedy'),
        ('Drama', 'Drama'),
        ('Horror', 'Horror'),
        ('Sci-Fi', 'Sci-Fi'),
        ('Romance', 'Romance'),
        ('Animation', 'Animation'),
        ('Thriller', 'Thriller'),
        ('Documentary', 'Documentary'),
    ]

    title = models.CharField(max_length=255)
    description = models.TextField()
    trailer_url = models.URLField(validators=[URLValidator()], verbose_name="Trailer URL")
    genre = models.CharField(max_length=50, choices=GENRE_CHOICES)
    release_year = models.PositiveIntegerField(
        validators=[MinValueValidator(1900), MaxValueValidator(datetime.date.today().year)],
        help_text="Enter a valid year."
    )
    poster_image = models.ImageField(
        upload_to='img',
        validators=[FileExtensionValidator(allowed_extensions=['png', 'jpg', 'jpeg'])]
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def poster_preview(self):
        if self.poster_image:
            return mark_safe(f'<img src="{self.poster_image.url}" width="150" height="auto" />')
        return "No poster available"

    poster_preview.short_description = "Poster Preview"

    def __str__(self):
        return f"{self.title} ({self.release_year})"