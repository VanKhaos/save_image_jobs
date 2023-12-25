from hmac import new
import os
import re
import sys
import json
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import colorama
from colorama import init, Fore, Back, Style
import numpy as np
import locale
from datetime import datetime
from pathlib import Path
import folder_paths

colorama.init(autoreset=True)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'comfy'))
original_locale = locale.setlocale(locale.LC_TIME, '')


class SaveImageJobs:
	def __init__(self):
		self.output_dir = folder_paths.get_output_directory()
		self.type = 'output'
		self.prefix_append = ''

	@classmethod
	def INPUT_TYPES(s):
		return {
			'required': {
				'images': ('IMAGE', ),
			},
			'hidden': {'prompt': 'PROMPT', 'extra_pnginfo': 'EXTRA_PNGINFO'},
		}

	RETURN_TYPES = ()
	FUNCTION = 'save_images_job'
	OUTPUT_NODE = True
	CATEGORY = 'image'

	def get_subfolder_path(self, image_path, output_path):
		image_path = Path(image_path).resolve()
		output_path = Path(output_path).resolve()
		relative_path = image_path.relative_to(output_path)
		subfolder_path = relative_path.parent

		return str(subfolder_path)

	# Get current counter number from file names
	def get_latest_counter(self, one_counter_per_folder, folder_path, filename_prefix, counter_digits, counter_position='last'):
		counter = 1
		if not os.path.exists(folder_path):
			print(f"Folder {folder_path} does not exist, starting counter at 1.")
			return counter

		try:
			files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
			if files:
				counters = [int(f[-(4 + counter_digits):-4]) if f[-(4 + counter_digits):-4].isdigit() else 0 for f in files if one_counter_per_folder == 'enabled' or f.startswith(filename_prefix)]

				if counters:
					counter = max(counters) + 1

		except Exception as e:
			print(f"An error occurred while finding the latest counter: {e}")

		return counter

	@staticmethod
	def find_keys_recursively(d, keys_to_find, found_values):
		for key, value in d.items():
			if key in keys_to_find:
				found_values[key] = value
			if isinstance(value, dict):
				SaveImageJobs.find_keys_recursively(value, keys_to_find, found_values)

	@staticmethod
	def remove_file_extension(value):
		if isinstance(value, str) and value.endswith('.safetensors'):
			base_value = os.path.basename(value)
			value = base_value[:-12]
		if isinstance(value, str) and value.endswith('.pt'):
			base_value = os.path.basename(value)
			value = base_value[:-3]

		return value

	@staticmethod
	def find_parameter_values(target_keys, obj, found_values=None):
		if found_values is None:
			found_values = {}

		if not isinstance(target_keys, list):
			target_keys = [target_keys]

		loras_string = ''
		for key, value in obj.items():
			if 'loras' in target_keys:
				# Match both formats: lora_xx and lora_name_x
				if re.match(r'lora(_name)?(_\d+)?', key):
					if value.endswith('.safetensors'):
						value = SaveImageJobs.remove_file_extension(value)
					if value != 'None':
						loras_string += f'{value}, '

			if key in target_keys:
				if (isinstance(value, str) and value.endswith('.safetensors')) or (isinstance(value, str) and value.endswith('.pt')):
					value = SaveImageJobs.remove_file_extension(value)
				found_values[key] = value

			if isinstance(value, dict):
				SaveImageJobs.find_parameter_values(target_keys, value, found_values)

		if 'loras' in target_keys and loras_string:
			found_values['loras'] = loras_string.strip(', ')

		if len(target_keys) == 1:
			return found_values.get(target_keys[0], None)

		return found_values

	@staticmethod
	def generate_custom_name(keys_to_extract, prefix, delimiter_char, resolution, prompt):
		custom_name = prefix

		if prompt is not None and len(keys_to_extract) > 0:
			found_values = {'resolution': resolution}
			SaveImageJobs.find_keys_recursively(prompt, keys_to_extract, found_values)
			for key in keys_to_extract:
				value = found_values.get(key)
				if value is not None:
					if key == 'cfg' or key =='denoise':
						try:
							value = round(float(value), 1)
						except ValueError:
							pass

					if (isinstance(value, str) and value.endswith('.safetensors')) or (isinstance(value, str) and value.endswith('.pt')):
						value = SaveImageJobs.remove_file_extension(value)

					custom_name += f'{value}'

		return custom_name.strip(delimiter_char)

	@staticmethod
	def save_job_to_json(prompt, file_name, resolution, output_path):
		prompt_keys_to_save = {}

		# Get filename data
		prompt_keys_to_save['filename'] = file_name
		
		# Get resolution data
		prompt_keys_to_save['resolution'] = resolution

		# Get model data		
		models = SaveImageJobs.find_parameter_values(['ckpt_name','vae_name', 'model_name', 'clip_skip', 'empty_latent_width', 'empty_latent_height'], prompt)
		if models.get('ckpt_name'):
			prompt_keys_to_save['checkpoint'] = models['ckpt_name']
		if models.get('vae_name'):
			prompt_keys_to_save['vae'] = models['vae_name']
		if models.get('model_name'):
			prompt_keys_to_save['upscale_model'] = models['model_name']
		if models.get('clip_skip'):
			prompt_keys_to_save['clip_skip'] = models['clip_skip']
		if models.get('empty_latent_width'):
			prompt_keys_to_save['empty_latent_width'] = models['empty_latent_width']
		if models.get('empty_latent_height'):
			prompt_keys_to_save['empty_latent_height'] = models['empty_latent_height']
		

		# Get sampler data
		prompt_keys_to_save['sampler_parameters'] = SaveImageJobs.find_parameter_values(['steps', 'cfg', 'sampler_name', 'scheduler', 'denoise'], prompt)
		
		# Get promt data
		if prompt is not None:
			#print(prompt)
			for key in prompt:
				class_type = prompt[key].get('class_type', None)
				inputs = prompt[key].get('inputs', {})

				# Seed prompt structure
				if class_type == 'Seed Everywhere':
					prompt_keys_to_save['seed'] = inputs.get('seed')

				# Upscale prompt structure
				if class_type == 'LatentUpscale':
					prompt_keys_to_save['upscale_method'] = inputs.get('upscale_method')
					prompt_keys_to_save['upscale_width'] = inputs.get('width')
					prompt_keys_to_save['upscale_height'] = inputs.get('height')
					prompt_keys_to_save['upscale_crop'] = inputs.get('crop')
				if class_type == 'KSamplerAdvanced':
					prompt_keys_to_save['upscale_sampler'] = 'KSamplerAdvanced'
					prompt_keys_to_save['upscale_sampler_steps'] = inputs.get('steps')
					prompt_keys_to_save['upscale_sampler_cfg'] = inputs.get('cfg')
					prompt_keys_to_save['upscale_sampler_name'] = inputs.get('sampler_name')
					prompt_keys_to_save['upscale_sampler_scheduler'] = inputs.get('scheduler')
					prompt_keys_to_save['upscale_sampler_start_at_step'] = inputs.get('start_at_step')
					prompt_keys_to_save['upscale_sampler_end_at_step'] = inputs.get('end_at_step')
					prompt_keys_to_save['upscale_sampler_add_noise'] = inputs.get('add_noise')					
				
				# LoRA Stacker prompt structure
				if class_type == 'LoRA Stacker':
					prompt_keys_to_save['lora_stacker'] = 'LoRA Stacker'
					prompt_keys_to_save['lora_stacker_mode'] = inputs.get('input_mode')
					prompt_keys_to_save['lora_stacker_count'] = inputs.get('lora_count')
					lora_count = int(inputs.get('lora_count'))
					for i in range(lora_count):
						lora_name = SaveImageJobs.remove_file_extension(inputs.get(f'lora_name_{i+1}'))
						prompt_keys_to_save[f'lora_stacker_lora_name_{i+1}'] = lora_name
						prompt_keys_to_save[f'lora_stacker_lora_wt_{i+1}'] = inputs.get(f'lora_wt_{i+1}')
						prompt_keys_to_save[f'lora_stacker_lora_model_str_{i+1}'] = inputs.get(f'model_str_{i+1}')
						prompt_keys_to_save[f'lora_stacker_lora_clip_str_{i+1}'] = inputs.get(f'clip_str_{i+1}')

				# Efficiency Loaders prompt structure
				if class_type == 'Efficient Loader' or class_type == 'Eff. Loader SDXL':
					if 'positive' in inputs and 'negative' in inputs:
						prompt_keys_to_save['positive_prompt'] = inputs.get('positive')
						prompt_keys_to_save['negative_prompt'] = inputs.get('negative')
				# KSampler/UltimateSDUpscale prompt structure
				elif class_type == 'KSampler' or class_type == 'KSamplerAdvanced' or class_type == 'UltimateSDUpscale':
					positive_ref = inputs.get('positive', [])[0] if 'positive' in inputs else None
					negative_ref = inputs.get('negative', [])[0] if 'negative' in inputs else None
				
					positive_text = prompt.get(str(positive_ref), {}).get('inputs', {}).get('text', None)
					negative_text = prompt.get(str(negative_ref), {}).get('inputs', {}).get('text', None)
				
					# If we get non text inputs
					if positive_text is not None:
						if isinstance(positive_text, list):
							if len(positive_text) == 2:
								if isinstance(positive_text[0], str) and len(positive_text[0]) < 6:
									if isinstance(positive_text[1], (int, float)):
										continue
						prompt_keys_to_save['positive_prompt'] = positive_text

					if negative_text is not None:
						if isinstance(negative_text, list):
							if len(negative_text) == 2:
								if isinstance(negative_text[0], str) and len(negative_text[0]) < 6:
									if isinstance(negative_text[1], (int, float)):
										continue
						prompt_keys_to_save['positive_prompt'] = negative_text

		# Append data and save
		json_file_path = os.path.join(output_path, 'jobs.json')
		existing_data = {}
		if os.path.exists(json_file_path):
			try:
				with open(json_file_path, 'r') as f:
					existing_data = json.load(f)
			except json.JSONDecodeError:
				print(f"The file {json_file_path} is empty or malformed. Initializing with empty data.")
				existing_data = {}

		timestamp = datetime.now().strftime('%d.%m.%Y | %H:%M:%S:%f')
		new_entry = {}
		new_entry[timestamp] = prompt_keys_to_save
		existing_data.update(new_entry)

		with open(json_file_path, 'w') as f:
			json.dump(existing_data, f, indent=4)


	def save_images_job(self,
				 images,
				 counter_digits=4,
				 counter_position='last',
				 one_counter_per_folder='enabled',
				 delimiter='',
				 filename_keys='',
				 foldername_keys='',
				 save_metadata='enabled',
				 filename_prefix='',
				 foldername_prefix=datetime.now().strftime('%Y-%m-%d'),
				 extra_pnginfo=None,
				 prompt=None
				):

		delimiter_char = "" if delimiter =='underscore' else '' if delimiter =='dot' else ''

		# Get set resolution value
		i = 255. * images[0].cpu().numpy()
		img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
		resolution = f'{img.width}x{img.height}'

		filename_keys_to_extract = [item.strip() for item in filename_keys.split(',')]
		foldername_keys_to_extract = [item.strip() for item in foldername_keys.split(',')]

		custom_filename = SaveImageJobs.generate_custom_name(filename_keys_to_extract, filename_prefix ,delimiter_char, resolution, prompt)
		custom_foldername = SaveImageJobs.generate_custom_name(foldername_keys_to_extract, foldername_prefix, delimiter_char, resolution, prompt)

		# Create and save images
		try:
			full_output_folder, filename, _, _, custom_filename = folder_paths.get_save_image_path(custom_filename, self.output_dir, images[0].shape[1], images[0].shape[0])
			output_path = os.path.join(full_output_folder, custom_foldername)
			os.makedirs(output_path, exist_ok=True)
			counter = self.get_latest_counter(one_counter_per_folder, output_path, filename, counter_digits, counter_position)
			for image in images:
				i = 255. * image.cpu().numpy()
				img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
				if save_metadata == 'enabled':
					metadata = PngInfo()
					if prompt is not None:
						metadata.add_text('prompt', json.dumps(prompt))
					if extra_pnginfo is not None:
						for x in extra_pnginfo:
							metadata.add_text(x, json.dumps(extra_pnginfo[x]))

				filename = f'{counter:0{counter_digits}}.png'
				file_name = f'{counter:0{counter_digits}}.png'
				print(Fore.GREEN + f'Saving Jobdata for {file_name} to {output_path}\jobs.json')

				# Save Image Job Data
				SaveImageJobs.save_job_to_json(prompt, file_name, resolution, output_path)           
				counter += 1
	
		except OSError as e:
			print(f'An error occurred while creating the subfolder or saving the image: {e}')
		else:
			results = list()
			return { 'ui': { 'images': results } }


NODE_CLASS_MAPPINGS = {
    'Save Image Jobs': SaveImageJobs,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'Save Image Jobs': 'Save Image Jobs',
}
