import keras
from pickle import load
from pickle import dump
from numpy import array
from numpy import argmax
from keras.utils import plot_model
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
from nltk.translate.bleu_score import corpus_bleu
from keras.preprocessing.sequence import pad_sequences


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


def create_all_sequences(tokenizer, max_length, descriptions, photos, vocab_size):
	X1, X2, y = list(), list(), list()
	# walk through each image identifier
	for key, desc_list in descriptions.items():
		# walk through each description for the image
		for desc in desc_list:
			# encode the sequence
			seq = tokenizer.texts_to_sequences([desc])[0]
			# split one sequence into multiple X,y pairs
			for i in range(1, len(seq)):
				# split into input and output pair
				in_seq, out_seq = seq[:i], seq[i]
				# pad input sequence
				in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
				# encode output sequence
				out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
				# store
				X1.append(photos[key][0])
				X2.append(in_seq)
				y.append(out_seq)
	return array(X1), array(X2), array(y)


def define_model(vocab_size, max_length):
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
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	# summarize model
	print(model.summary())
	plot_model(model, to_file='model.png', show_shapes=True)
	return model

def word_for_id(integer, tokenizer):
	"""
	Mapea la predicción que es un entero 
	a una palabra del tokenizer
	"""
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

def generate_desc(model, tokenizer, photo, max_length):
	"""
	Genera una descripcion textual dado un modelo entrenado
	y una foto preparada como input
	"""	
	in_text = 'startseq'	
	for i in range(max_length):
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		sequence = pad_sequences([sequence], maxlen=max_length)
		yhat = model.predict([photo,sequence], verbose=0)
		yhat = argmax(yhat)
		word = word_for_id(yhat, tokenizer)
		if word is None:
			break
		in_text += ' ' + word
		if word == 'endseq':
			break
	return in_text

def evaluate_model(model, descriptions, photos, tokenizer, max_length):
	"""
	Métricas del modelo
	"""
	actual, predicted = list(), list()
	# se va a iterar sobre todas las descripciones de prueba
	for key, desc_list in descriptions.items():
		# se genera la descipción
		yhat = generate_desc(model, tokenizer, photos[key], max_length)
		# se almacena la descripción real y la de predicha por el modelo
		references = [d.split() for d in desc_list]
		actual.append(references)
		predicted.append(yhat.split())
	# Se calcula el BLUE score
	print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


def data_generator(descriptions, photos, tokenizer, max_length, vocab_size):
	"""

	"""
	while 1:
		for key, desc_list in descriptions.items():
			photo = photos[key][0]
			in_img, in_seq, out_word = create_sequences(tokenizer, max_length, 
														desc_list, photo, 
														vocab_size)
			yield [[in_img, in_seq], out_word]


def loading_train_model(model, epochs, train_descriptions, train_features, tokenizer,
				max_length, vocab_size, val_descriptions, val_features):
	"""
	Se entrena el modelo con el datagenerator
	"""
	steps = len(train_descriptions)
	val_steps = len(val_descriptions)

	es = keras.callbacks.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, 
												patience=2, verbose=0, mode='auto', 
												baseline=None, restore_best_weights=False)

	for i in range(epochs):
		generator = data_generator(train_descriptions, train_features, 
								tokenizer, max_length, vocab_size)
		
		val_generator = data_generator(val_descriptions, val_features,
									tokenizer, max_length, vocab_size)
		model.fit_generator(generator,
							steps_per_epoch = steps,
							validation_data = val_generator,
							validation_steps = val_steps,
							epochs = 1,  verbose = 1, 
							callbacks = [es], shuffle = True)
		model.save('model_' + str(i) + '.h5')



def train_model(tokenizer, max_length, train_descriptions, train_features, vocab_size, test_descriptions, test_features):
	
	X1train, X2train, ytrain = create_all_sequences(tokenizer, max_length, train_descriptions, train_features, vocab_size)
	X1test, X2test, ytest = create_all_sequences(tokenizer, max_length, test_descriptions, test_features, vocab_size)

	model = define_model(vocab_size, max_length)
	# define checkpoint callback
	filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
	checkpoint = keras.callbacks.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
	# fit model
	model.fit([X1train, X2train], ytrain, epochs=20, verbose=1, callbacks=[checkpoint], validation_data=([X1test, X2test], ytest))