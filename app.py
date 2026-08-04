"""Hugging Face Spaces entry point for ShelfHeat."""

from shelfheat.webapp import create_demo

demo = create_demo()

if __name__ == "__main__":
    demo.launch()
