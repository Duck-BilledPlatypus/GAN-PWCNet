T2netModel
	initialize
		BaseModel
	set_input
	forward
	backward_D_basic
	backward_D_image
	backward_D_feature
	foreward_G_basic
	backward_synthesis2real
	backward_translated2depth
	backward_real2depth
	optimize_parameters
	validation_target
	
BaseModel
	initialize
	set_input
	update_learning_rate
	get_current_errors
	get_current_visuals
	save_networks
	load_networks

TNetModel
	initialize
	set_input
	forward
	foreward_G_basic
	backward_task
	optimize_parameters
	validation_target

network
	get_norm_layer
	get_nonlinearity_layer
	get_scheduler
	init_weights
	print_network
	init_net
	_freeze
	_unfreeze
	define_G
	define_D
	define_featureD

	class GaussianNoiseLayer
	class _InceptionBlock
	class _EncoderBlock
	class _DownBlock
	class _ShuffleUpBlock
	class _DecoderUpBlock
	class _OutputBlock

	class _ResGenerator
	class _PreUNet16
	class _UNetGenerator
	class _MultiscaleDiscriminator
	class _Discriminator
	class _FeatureDiscriminator

