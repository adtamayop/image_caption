import keras
from pickle import dump
from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint

def load_doc(filename):
	"""
	Carga y retorna archivo de texto
	"""
	file = open(filename, 'r')	
	text = file.read()
	file.close()
	return text

def load_set(filename):
	"""
	Carga una lista predefinidas de datos
	(Entrenamiento, validacion, prueba)
	"""
	doc = load_doc(filename)
	dataset = list()
	for line in doc.split('\n'):
		if len(line) < 1:
			continue
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return set(dataset)

def load_clean_descriptions(filename, dataset):
	"""
	Carga las descripciones limpias del archivo de descripciones,
	y añade 'startseq' y 'endseq' al inicio y fin de la descripcion
	"""
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):		
		tokens = line.split()		
		image_id, image_desc = tokens[0], tokens[1:]		
		if image_id in dataset:		
			if image_id not in descriptions:
				descriptions[image_id] = list()		
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'		
			descriptions[image_id].append(desc)
	return descriptions

def load_photo_features(filename, dataset):
	"""
	Carga todo el archivo de features  y selecciona 
	el conjunto correspondiente (train,val,test)
	"""
	all_features = load(open(filename, 'rb'))
	features = {k: all_features[k] for k in dataset}
	return features


def to_lines(descriptions):
	"""
	Convierte un diccionario limpio de 
	descripciones a una lista de descripciones
	"""
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc


def create_tokenizer(descriptions):
	"""
	Codifica lista de descripciones para crear un 
	text corpus, y retorna el tokenizer ajustado
	"""
	lines = to_lines(descriptions)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer


def max_length(descriptions):
	"""
	Se calcula el tamaño máximo de 
	las descripción con más palabras
	"""
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)

def create_sequences(tokenizer, max_length, desc_list, 
					 photo, vocab_size):
	"""
	crea secuencias de imagenes, secuencias de entrada 
	y palabras de salida para una imagen dada	
	"""
	X1, X2, y = list(), list(), list()	
	for desc in desc_list:
		# codifica la secuencia de texto
		seq = tokenizer.texts_to_sequences([desc])[0]
		# parte la secuencia en multiples pares X,y
		for i in range(1, len(seq)):
			# parte en un par de entrada y salida
			in_seq, out_seq = seq[:i], seq[i]
			# aplica el pad_sequences a la secuencia de entrada
			in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
			# codifica la secuencia de salida
			out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
			X1.append(photo)
			X2.append(in_seq)
			y.append(out_seq)
	return array(X1), array(X2), array(y)


def define_model(vocab_size, max_length):

	adam = keras.optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999, amsgrad=False)
	# feature extractor model
	inputs1 = Input(shape=(4096,))
	fe1 = Dropout(0.5)(inputs1)
	fe2 = Dense(256, activation='relu')(fe1)
	# sequence model
	inputs2 = Input(shape=(max_length,))
	se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
	se2 = Dropout(0.5)(se1)
	se3 = LSTM(256)(se2)
	# decoder model
	decoder1 = add([fe2, se3])
	decoder2 = Dense(256, activation='relu')(decoder1)
	outputs = Dense(vocab_size, activation='softmax')(decoder2)
	# tie it together [image, seq] [word]
	model = Model(inputs=[inputs1, inputs2], outputs=outputs)
	# compile model
	model.compile(loss='categorical_crossentropy', optimizer=adam)
	# summarize model
	# model.summary()
	plot_model(model, to_file='model.png', show_shapes=True)
	return model

# data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions, photos, tokenizer, max_length, vocab_size):
	# loop for ever over images
	while 1:
		for key, desc_list in descriptions.items():
			# retrieve the photo feature
			photo = photos[key][0]
			in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, photo, vocab_size)
			yield [[in_img, in_seq], out_word]

# load training dataset (6K)
filename = 'C:/Users/atamayop/Desktop/image_caption/data/Flickr_8k.trainImages.txt'
# filename = '.../data/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions('C:/Users/atamayop/Desktop/image_caption/src/files/descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))

# photo features
train_features = load_photo_features('C:/Users/atamayop/Desktop/image_caption/src/files/features.pkl', train)
print('Photos: train=%d' % len(train_features))

# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
dump(tokenizer, open('tokenizer.pkl', 'wb'))

vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)

# define the model
model = define_model(vocab_size, max_length)
# train the model, run epochs manually and save after each epoch
epochs = 2
steps = len(train_descriptions)


es = keras.callbacks.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto', baseline=None, restore_best_weights=False)

for i in range(epochs):
	# create the data generator
	generator = data_generator(train_descriptions, train_features, tokenizer, max_length, vocab_size)
	# fit for one epoch
	model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1, callbacks=[es], shuffle=True)
	# save model
	model.save('model_' + str(i) + '.h5')