class HParams():
	def __init__(self):
		self.hop=192
		self.sr=16000
		self.min_level_db=-100
		self.ref_level_db=20

		self.gen_units=128
		self.siam_units=128
		self.disc_units=512

		self.shape=24
		self.vec_len=128
		self.bs=32
		self.delta=2.
		self.n_save=5
		self.save_path='./checkpoints'

		self.epochs=2000
		self.lr=0.0001
		self.opt_alpha=0.5
		self.gloss_delta=1.
		self.mloss_delta=10.
		self.idloss_delta=0.5
		self.use_id=False
		self.use_lstm=False
		self.gupt=3

hparams = HParams()