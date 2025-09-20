######################
Prediction of Methane Adsorption Isotherms in Metal-organic Frameworks by Neural Networks: Two-dimensional Energy Gradient Feature and Masked Learning Mechanism

--------------------------------
The neural network architecture consists of one input layer, six hidden layers and one output layer. The number of neurons in each layer is as follows: 64-128-256-256-128-64。
Input features：2D-EGF  LCD  	PLD	desity(g/cm^3)	VSA(m^2/cm^3)	GSA(m^2/g)	Vp(cm^3/g)	void_fraction	functional_groups	metal_linker	organic_linker1	organic_linker2	topology	 H	C	N	O	F	P	S	Cl	Br	I	V	Cu	Zn	Cr	Ni	Ba	metal type	 total degree of unsaturation	metalic percentage	 oxygetn-to-metal ratio	electronegtive-to-total ratio	 weighted electronegativity per atom	 nitrogen to oxygen	uc_volume	alpha	beta	gamma	lengtha	lengthb	lengthc 	TK
Output targets：bi_0   qmax	delta_H	0.001	5	10	15	20	25	30	35	40	45	50	55	60	65	70	75	80	85	90	95	99.99999bar  (0-99.99bar)
Activation functions：Tanh
Learning rate：0.001
optimization algorithm：Adam
Loss function: MSEloss
training epochs:15000



