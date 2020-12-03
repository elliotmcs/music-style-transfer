import wandb

def set_param(name, default):
	val = default
	try:
		val = eval("wandb.config.{}".format(name))
	except AttributeError:
		exec("wandb.config.{0}={1}".format(name,default if type(default)!=str else "\"{}\"".format(default)))
		pass
	return val

class HParams():
	def __init__(self):
		self.hop=set_param("hop",192)
		self.sr=set_param("sr", 16000)
		self.min_level_db=set_param("min_level_db",-100)
		self.ref_level_db=set_param("ref_level_db",20)

		self.gen_units=set_param("gen_units", 128)
		self.siam_units=set_param("siam_units", 128)
		self.disc_units=set_param("disc_units", 512)

		self.shape=set_param("shape",24)
		self.vec_len=set_param("vec_len",128)
		self.bs=set_param("bs",32)
		self.delta=set_param("delta",2.)
		self.n_save=set_param("n_save",5)
		self.save_path=set_param("save_path",'./checkpoints')

		self.epochs=set_param("epochs",2000)
		self.lr=set_param("lr", 0.0001)
		self.opt_alpha=set_param("opt_alpha",0.5)
		self.gloss_delta=set_param("gloss_delta",1.)
		self.mloss_delta=set_param("mloss_delta",10.)
		self.idloss_delta=set_param("idloss_delta",0.5)
		self.use_id=set_param("use_id",False)
		self.use_lstm=set_param("use_lstm",False)
		self.gupt=set_param("gupt",3)

hparams = HParams()