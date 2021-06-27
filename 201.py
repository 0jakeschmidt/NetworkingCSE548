# To l o a d a d a t a s e t f i l e in Python , you can use Pandas . Impor t pandas

import pandas as pd
# Impor t numpy t o per form o p e r a t i o n s on t h e d a t a s e t
import numpy as np



dataset = pd.read_csv(./NSL−KDD/KDDTrain +.txt , header=None)
X = dataset.iloc[:, 0:−2].values
label_column = dataset.iloc[:, −2].values
y = []
for i in range(len( label_column )):
    if label_column[i] == 'normal':
        y.append(0)
    else:
        y.append(1)
# Conver t i−s t t o a r r ay
y = np.array(y)



from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(
# The column numbers t o be t r an s f o rme d ( [ 1 , 2 , 3 ] r e p r e s e n t s t h r e e columns t o be t r a n s f e r r e d )
[('one_hot_encoder ', OneHotEncoder(), [1 ,2 ,3])],
# Leave t h e r e s t o f t h e columns un touched
remainder ='passthrough'
)
X = np.array(ct.fit_transform(X), dtype=np.float)

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split (X, y, test_size =0.25 , random_state = 0)




from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# S c a l i n g t o t h e range [ 0 , 1 ]
X_train = sc.fit_transform( X_train )
X_test = sc.fit_transform( X_test )



from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(units = 6, kernel_initializer = 'uniform',
activation = 'relu', input_dim = len( X_train[0])))
# Adding t h e second h i d den l a y e r , 6 nodes
classifier.add(Dense(units = 6, kernel_initializer = 'uniform',
activation = 'relu'))
# Adding t h e o u t p u t l a y e r , 1 node ,
# s igm o i d on t h e o u t p u t l a y e r i s t o en s u re t h e ne twork o u t p u t i s
#be tween 0 and 1
classifier.add(Dense(units = 1, kernel_initializer = 'uniform',
activation = 'sigmoid'))



classifier.compile( optimizer = 'adam', loss = 'binary_crossentropy',metrics = ['accuracy'])
# run t r a i n i n g w i t h b a t c h s i z e e q u a l s t o 10 and 10 e p oc h s w i l l be
#per formed
classifierHistory = classifier.fit( X_train , y_train , batch_size = 10,epochs = 10)

loss , accuracy = classifier.evaluate( X_train , y_train )
print('Print the loss and the accuracy of the model on the dataset')
print('Loss [0 ,1]: %.4f' % (loss), 'Accuracy [0 ,1]: %.4f' % (accuracy))


# u s i n g t h e t r a i n e d model t o p r e d i c t t h e Tes t d a t a s e t r e s u l t s
y_pred = classifier.predict(X_test)
# y p r e d e q u a l s t o 0 i f t h e p r e d i c t i o n i s l e s s than 0 . 9 or e q u a l t o
#0 . 9 ,
# y p r e d e q u a l s t o 1 i f t h e p r e d i c a t i o n i s g r e a t e r than 0 . 9
y_pred = (y_pred > 0.9)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix ( y_test , y_pred )

print('Print the Confusion Matrix:')
print(cm)


plt.plot( classifierHistory.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend (['train'], loc='upper left')
plt.savefig('accuracy sample.png')
plt.show ()


print('Plot the loss')
plt.plot( classifierHistory.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend (['train'], loc='upper left')
plt.savefig('loss sample.png')
plt.show ()
