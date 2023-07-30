# rectangle_area/tests.py

from django.test import TestCase
from .models import Rectangle

class RectangleModelTests(TestCase):
    def test_get_area(self):
        rectangle = Rectangle.objects.create(length=5, width=3)
        self.assertEqual(rectangle.get_area(), 15)
