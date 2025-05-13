"""
Vision Language Model (VLM) Runner Module

This module handles the execution of the Molmo VLM model for identifying
objects in satellite imagery.
"""

from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import os
import logging
from typing import Optional
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VLMRunner:
    """Class to handle VLM model loading and inference."""
    
    def __init__(self, model_name: str = 'cyan2k/molmo-7B-O-bnb-4bit'):
        """
        Initialize the VLM runner.
        
        Args:
            model_name: Name of the model to load from HuggingFace
        """
        self.model_name = model_name
        self.processor = None
        self.model = None
        
    def load_model(self) -> None:
        """
        Load the VLM model and processor.
        
        Raises:
            RuntimeError: If CUDA is not available
            Exception: For other model loading errors
        """
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available. This model requires GPU acceleration.")
                
            logger.info(f"Loading model: {self.model_name}")
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype='auto',
                device_map='auto'
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype='auto',
                device_map='auto'
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
            
    def process_image(self, image_path: str, query: str) -> Optional[str]:
        """
        Process a single image with the VLM model.
        
        Args:
            image_path: Path to the image file
            query: Text query for the model
            
        Returns:
            Generated text from the model, or None if processing fails
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            Exception: For other processing errors
        """
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
                
            if self.processor is None or self.model is None:
                self.load_model()
                
            # Process image and text
            inputs = self.processor.process(
                images=[Image.open(image_path)],
                text=query
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}
            
            # Generate output
            output = self.model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=2000, stop_strings="<|endoftext|>"),
                tokenizer=self.processor.tokenizer
            )
            
            # Decode output
            generated_tokens = output[0, inputs['input_ids'].size(1):]
            generated_text = self.processor.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            )
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return None

def run() -> None:
    """
    Main function to run VLM processing on all images in the dataset.
    """
    try:
        # Get number of images in dataset
        image_dir = 'benchmark-UAV-VLPA-nano-30/images'
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
            
        num_samples = len([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        logger.info(f"Processing {num_samples} images")
        
        # Initialize VLM runner
        vlm_runner = VLMRunner()
        
        # Process all images
        with open('identified_points.txt', 'w') as f:
            for i in range(1, num_samples + 1):
                logger.info(f"Processing image {i}/{num_samples}")
                
                image_path = os.path.join(image_dir, f"{i}.jpg")
                query = "This is the satellite image of a city. Please, point all the buildings."
                
                generated_text = vlm_runner.process_image(image_path, query)
                
                if generated_text:
                    f.write(f"{i}, {generated_text}\n")
                    logger.debug(f"Generated text for image {i}: {generated_text}")
                else:
                    logger.warning(f"Failed to process image {i}")
                    
        logger.info("Processing complete")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    run()