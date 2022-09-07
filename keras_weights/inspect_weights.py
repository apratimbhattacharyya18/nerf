import tensorflow as tf

model = tf.keras.models.load_model('./model-0050',compile=False)

#print('attributes ',[wt.name for wt in model.weights])#dir(model), model.near_depth,  model.far_depth, model.num_samples_per_ray, model.num_fine_samples_per_ray,
'''print(
    model.sampling_method,
    model.jitter,
    model.background_color,
    model.last_bin_width,
    model._scene_box
    )'''

print(model)

for i in range(len(model._coarse_scene_mlp.hidden_layers)):
	print('_coarse_scene_mlp.hidden_layers ',i,model._coarse_scene_mlp.hidden_layers[i].weights[0].shape,
		model._coarse_scene_mlp.hidden_layers[i].weights[1].shape)

print('viewdir_bottleneck_layer ',i,model._coarse_scene_mlp.viewdir_bottleneck_layer.weights[0].shape,
		model._coarse_scene_mlp.viewdir_bottleneck_layer.weights[1].shape)

for i in range(len(model._coarse_scene_mlp.viewdir_hidden_layers)):
	print('_coarse_scene_mlp.viewdir_hidden_layers ',i,model._coarse_scene_mlp.viewdir_hidden_layers[i].weights[0].shape,
		model._coarse_scene_mlp.viewdir_hidden_layers[i].weights[1].shape)

print('density_layer ',i,model._coarse_scene_mlp.density_layer.weights[0].shape,
		model._coarse_scene_mlp.density_layer.weights[1].shape)

print('color_layer ',i,model._coarse_scene_mlp.color_layer.weights[0].shape,
		model._coarse_scene_mlp.color_layer.weights[1].shape)