"""Coverage-aware astrophotography stacking."""

from .models import ProcessingOptions, StackMode, StackResult
from .pipeline import stack_images

__all__ = ["ProcessingOptions", "StackMode", "StackResult", "stack_images"]

