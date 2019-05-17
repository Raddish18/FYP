import kivy
kivy.require('1.9.0')
 
from kivy.uix.boxlayout import BoxLayout 
from kivy.uix.label import Label 
from kivy.uix.button import Button 
from kivy.lang import Builder
from kivy.app import App 
from kivy.clock import Clock
from kivy.clock import mainthread
from kivy.uix.textinput import TextInput
from kivy.uix.image import Image 
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.filechooser import FileChooserListView
#from jnius import autoclass 


#Google Imports
import gspread 
from oauth2client.service_account import ServiceAccountCredentials
#import httplib2
import httplib
import ssl

import matplotlib
import _multibytecodec
import csv
import sklearn as sk
import pandas as pd
import numpy as np
from collections import Counter
import json

#SKLearn Stuff
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
#from sklearn.model_selection import train_test_split


from pathlib import Path
#from __future__ import print_function
# matplotlib for displaying the output
import matplotlib.pyplot as plt
import matplotlib.style as ms
ms.use('seaborn-muted')
#%matplotlib inline

# and IPython.display for audio output
import IPython.display

# Librosa for audio
import librosa
# And the display module for visualization
import librosa.display

import os

#Handle asking for permissions at runtime


Builder.load_string("""
<ScreenOne>:
    BoxLayout:
        orientation: "vertical"

        Label:
            text: "Genrefy"
            font_size: 40
        Button:
            text: "Choose a song from your files!"
            on_press:
                root.manager.transition.direction = 'left'    
                root.manager.transition.duration = 1                            
                root.manager.current = 'file_screen'
                #root.startRecording_clock()
                #root.classifier()

<FileChooseScreen>:
    BoxLayout:
        orientation: "vertical"

        FileChooserListView:
            id: filechoose
            on_selection: 
                root.classifier(filechoose.path, filechoose.selection)
                root.manager.current = 'screen_two'


<ScreenTwo>:
    BoxLayout:
        orientation: "vertical"

        Label:
            text: "Genrefy"

        Image:
            id: 'graph_image'
            size: 2000,500
            size_hint: None, None
            #source: "/home/cian/FYP/Kivy/ScreenTest/testplot.png"
            #on_touch_down: reload

        Button:
            text: "Go Back"
            #size: 200, 75
            #size_hint: None, None
            on_press:
                root.manager.transition.direction = 'right'
                root.manager.transition.duration = 1                            
                root.manager.current = 'screen_one'

""")



class ScreenOne(Screen):
    #pass
    #@mainthread
    def __init__(self, **kwargs):
        super(Screen, self).__init__(**kwargs)
	
	self.start = Button(text="Start Rec")
        self.start.bind(on_press=self.startRecording_clock)

    def startRecording_clock(self):
	#pass
	#self.start = self.ids["start_button"]
	self.start.disabled = True
	print("RECORDING START") 
        self.zero = 1 
	#Set duration of recording
        self.duration = 10
        #Clock.schedule_interval(self.updateDisplay, 1)
        #Clock.schedule_once(self.startRecording)
        #Clock.schedule_once(self.classifier)

    


class FileChooseScreen(Screen):
#"home/cian/FYP/Kivy/ScreenTest/testplot.png"

    def __init__(self, **kwargs):
        super(Screen, self).__init__(**kwargs)

        self.scope = ['https://www.googleapis.com/auth/drive']
        self.creds = ServiceAccountCredentials.from_json_keyfile_name('genrefy.json', self.scope)
        self.client = gspread.authorize(self.creds)

        self.sheet = self.client.open('music_data').sheet1

        filechooser = self.ids["filechoose"]
        filechooser.bind(on_selection=lambda x: self.classifier(self, filechooser.path, filechooser.selection))

    def classifier(self,path, choice):
	self.music_info = self.sheet.get_all_records()
	self.df = pd.DataFrame(self.music_info)

	self.song = str(choice[0]).encode('ascii', 'ignore')
        print "Analyzing: " + self.song
	#self.audio_path = self.path + self.song
        self.new_path = self.song
	y, sr = librosa.load(self.new_path, res_type='kaiser_fast')
	onset_env = librosa.onset.onset_strength(y, sr=sr)


	tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
	temp, beat = librosa.beat.beat_track(y=y, sr=sr, onset_envelope=onset_env)
	chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
	rmse = librosa.feature.rmse(y)
	spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
	spec_band = librosa.feature.spectral_bandwidth(y=y, sr=sr)
	rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
	z_x_r = librosa.feature.zero_crossing_rate(y)
	
	
	#print temp	
	chroma_stf =  np.mean(chroma_stft)
	rms  =  np.mean(rmse)
	spec_c =  np.mean(spec_cent)	
	spec_b = np.mean(spec_band)
	rollo = np.mean(rolloff)
	zero_cross = np.mean(z_x_r)
	genre = "blues"
	
	#name = "Blues Test"
	data = [[temp, chroma_stf, rms, spec_c, spec_b, rollo, zero_cross]]
	testDict = pd.DataFrame(data, columns=['tempo', 'chroma_stft', 'rmse', 'spectral_centroid', 'spectral_bandwidth', 'rolloff', 'z_cross_rate'] )
	#print testDict
	#print(testDict.shape)
	
	#print(np.unique(df['genre']))
	#print(df.groupby('genre').size())

	w = self.df['label']
    	X = self.df.drop(['label'], axis=1)
	#X.append(testDict)

    	#X_train, w_train  = train_test_split(X, w)
	X_test =  testDict
	w_test =  testDict


	print X.shape
	print X_test.shape
	
	scaler = StandardScaler()
	train_scaled = scaler.fit_transform(X)
	test_scaled = scaler.transform(X_test)

	mpls = ""
	rfs = ""
	dts = ""
	kns = ""
	nbs = ""


	mpl = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500, alpha=0.0001,
                     solver='sgd', verbose=10,  random_state=21,tol=0.000000001)	
	mpl.fit(train_scaled, w)
	print("Neural Network training accuracy",accuracy_score(w, mpl.predict(train_scaled)))
	w_pred = mpl.predict(test_scaled)
	mpls = w_pred[0]
	#mpls = mpls.encode('ascii', 'ignore')
	print(mpls)


	rf_model = RandomForestClassifier()
	rf_model.fit(train_scaled, w)
	print("Random Forest training accuracy",accuracy_score(w, rf_model.predict(train_scaled)))
	#print("Random Forest test accuracy",accuracy_score(w_test, rf_model.predict(test_scaled)))
	w_pred = rf_model.predict(test_scaled)
	rfs = w_pred[0]
	#rfs = rfs.encode('ascii', 'ignore')
	print(rfs)


	tree_model = DecisionTreeClassifier()
	tree_model.fit(X, w)
	print("Decision Tree training accuracy",accuracy_score(w, tree_model.predict(train_scaled)))
	#print("Decision Tree test accuracy",accuracy_score(w_test, tree_model.predict(test_scaled)))
	w_pred = tree_model.predict(test_scaled)
	dts = w_pred[0]
	#dts = dts.encode('ascii', 'ignore')
	print(dts)	


	knn = KNeighborsClassifier(n_neighbors=1)
	knn.fit(train_scaled, w)
	print("KNN training accuracy",accuracy_score(w, knn.predict(train_scaled)))
	#print("KNN test accuracy",accuracy_score(w_test, knn.predict(test_scaled)))
	w_pred = knn.predict(test_scaled)
	kns = w_pred[0]
	#kns = mpls.encode('ascii', 'ignore')
	print(kns)
	
	
	nb = BernoulliNB()
	nb.fit(train_scaled, w)
	print("Naive Bayes training accuracy",accuracy_score(w, nb.predict(train_scaled)))
	w_pred = nb.predict(test_scaled)
	nbs = w_pred[0]
	#nbs = nbs.encode('ascii', 'ignore')
	print(nbs)
	
	lab = [mpls, rfs, dts, kns, nbs]
	size = [] 
	names = []
	
	blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock = 0,0,0,0,0,0,0,0,0,0
	for name in lab:
		if name == 'blues':
			blues += 20
		elif name == 'classical':
			classical += 20
		elif name == 'country':
			country += 20
		elif name == 'disco':
			disco += 20
		elif name == 'hiphop':
			hiphop += 20
		elif name == 'jazz':
			jazz += 20
		elif name == 'metal':
			metal += 20
		elif name == 'pop':
			pop += 20
		elif name == 'reggae':
			reggae += 20
		elif name == 'rock':
			rock += 20
		names.append(name)
		
	size.extend([blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock])
	size = list(filter(lambda a: a != 0, size))
	print size
	names = sorted(names)
	print names
	name_sort = []
	for x in names:
    		if x not in name_sort:
      			name_sort.append(x)
	print name_sort
	

	fig1, ax1 = plt.subplots()
	ax1.pie(size, labels=name_sort, autopct='%1.1f%%',
        shadow=True, startangle=90)
	ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
	
	print("PLOTTING GRAPH")
        plt.suptitle(str(self.song), fontsize=16)
	plt.title("BPM: " + str(tempo), fontsize=10)
	plt.savefig('testplot.png')


class ScreenTwo(Screen):
#"home/cian/FYP/Kivy/ScreenTest/testplot.png"

    def __init__(self, **kwargs):
        super(Screen, self).__init__(**kwargs)
	
        graph = MyImage(source = "testplot.png")
        #self.ids['graph_image'] = self.add_widget(graph)
        self.ids.graph_image = self.add_widget(graph)

    def imageBuild(self):
        #self.image = self.ids['graph']
        #a = App.get_running_app()
        #a.graph_image.source = 'testplot.png'
        #Clock.schedule_interval(a.graph_image.reload(), 1)
	#return self.image
        pass

class MyImage(Image):

    #self.pos = root.x, root.top - self.height
    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            print("Reloading..")
            self.reload()


    #pass
 
 
# The ScreenManager controls moving between screens
screen_manager = ScreenManager()
 
# Add the screens to the manager and then supply a name
# that is used to switch screens
screen_manager.add_widget(ScreenOne(name="screen_one"))
screen_manager.add_widget(FileChooseScreen(name="file_screen"))
screen_manager.add_widget(ScreenTwo(name="screen_two"))
 
class KivyTut2App(App):
 
    def build(self):
        return screen_manager
 
sample_app = KivyTut2App()
sample_app.run()
