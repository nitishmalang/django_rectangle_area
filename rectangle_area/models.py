# rectangle_area/models.py

from django.db import models

class Rectangle(models.Model):
    length = models.DecimalField(max_digits=5, decimal_places=2)
    width = models.DecimalField(max_digits=5, decimal_places=2)

    def get_area(self):
        return self.length * self.width

    def __str__(self):
        return f"Rectangle (ID: {self.id})"
