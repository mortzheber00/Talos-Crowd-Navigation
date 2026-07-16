"""Shared floor styling (tiled, high-quality lighting) for walking and transit_hub."""

import mujoco


def setup_floor_ambience(spec):
	"""Apply the tiled floor texture, materials, and high-quality lighting to the
	simulator spec. Works on any spec with an existing 'floor' geom."""
	"""Skybox gradient, tiled floor material, sun + fill lighting."""
	# skybox: deep night blue at the zenith fading into a light horizon grey
	spec.add_texture(
		name='hub_skybox',
		type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
		builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
		rgb1=[0.06, 0.10, 0.24],   # zenith
		rgb2=[0.72, 0.76, 0.82],   # horizon
		width=512, height=3072)
	
	# floor: light tiles with dark grout lines -> strong, regular contrast
	# edges for the depth camera
	spec.add_texture(
		name='floor_tex',
		type=mujoco.mjtTexture.mjTEXTURE_2D,
		builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
		rgb1=[0.78, 0.80, 0.84],
		rgb2=[0.62, 0.66, 0.72],
		mark=mujoco.mjtMark.mjMARK_EDGE,
		markrgb=[0.35, 0.38, 0.42],  # soft grout lines (hard ones moire at distance)
		width=512, height=512)
	floor_mat = spec.add_material(
		name='floor_mat',
		texrepeat=[0.5, 0.5],   # with texuniform: 1 m tiles
		texuniform=True,
		reflectance=0.15,
		specular=0.3,
		shininess=0.4)
	floor_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = 'floor_tex'

	# apply material to floor geom
	for g in spec.geoms:
		if g.name == 'floor':
			g.material = 'floor_mat'
			g.rgba = [1.0, 1.0, 1.0, 1.0]  # untinted texture
			# render the plane as a finite 30x30 m platform (the
			# collision plane stays infinite regardless of the size)
			g.size = [15.0, 15.0, 1.0]
			break

	# enhance lighting: larger shadow map, dim the default headlight
	spec.stat.extent = 45.0

	# main sun: strong, low-angle directional light with soft shadows
	spec.worldbody.add_light(
		name='sun_main',
		type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
		diffuse=[0.75, 0.73, 0.68],
		specular=[0.3, 0.3, 0.3],
		pos=[0.0, 0.0, 12.0],
		dir=[0.35, -0.5, -1.0],
		castshadow=True)

	# soft fill from the opposite side so shadowed faces stay readable
	spec.worldbody.add_light(
		name='fill_light',
		type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
		diffuse=[0.18, 0.19, 0.22],
		specular=[0.0, 0.0, 0.0],
		pos=[0.0, 0.0, 8.0],
		dir=[-0.4, 0.5, -1.0],
		castshadow=False)

	# dim the default "sun" (replace its shadow with the custom directional lights above)
	for light in spec.lights:
		if light.name == 'sun':
			light.diffuse = [0.25, 0.25, 0.25]
			light.castshadow = False

	# high-res shadow map for soft, clean shadow edges; dim headlight
	spec.visual.quality.shadowsize = 8192
	spec.visual.headlight.diffuse = [0.25, 0.25, 0.25]
	spec.visual.headlight.ambient = [0.12, 0.12, 0.12]
