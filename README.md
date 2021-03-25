# Project-1: Text Data Analysis
# a.- Realización de análisis de sentimiento
0. [Insalamos la librería  textblob](#schema0)
1. [Importamos librerías](#schema1)
2. [Cargamos los datos](#schema2)
3. [Vamos hacer una prueba de análisis de una frase](#schema3)
4. [Vamos a comprobar si hay nulos y si hay nulos los eliminamos.](#schema4)
5. [Vamos a crear una lista con todas la polaridades de los comentarios y añadirlo al dataset como una columna nueva](schema5)
#  b.- Representación de Wordcloud de Sentimientos
6. [Creamos un dataset nuevo que solo contenga los valores de polarity = 1](#schema6)
7. [Instalamos WordCloud e importamos la librería](#schema7)
8. [Creamos la lista de `STOPWORDS` y hacemos que los comentarios es una cadena](#schema8)
9. [Creamos el tamaño de imagen](#schema9)
10. [Dibujamos la figura con el texto más positivo](#schema10)
11. [Dibujamos la figura con el texto más negativo](#schema11)

# c.- Analizar etiquetas de tendencias y vistas de Youtube
<hr>

<a name="schema0"></a>

# 0. Instalamos la librería textblob

Nosotros lo instalamos con conda porque estamos trabajando con un environment de conda
~~~python
conda install -c conda-forge textblob
~~~
Documentación

https://textblob.readthedocs.io/en/dev/



<hr>

<a name="schema1"></a>

# 1. Importamos librerías
~~~python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
~~~
<hr>

<a name="schema2"></a>

# 2. Cargamos los datos
Los datos que vamos a usar estan alojados aqui: https://drive.google.com/drive/u/0/folders/10owYwrtRQIRCawOFgZy1qY7gG3ta8VfS
Al cargarlo así nos da un error:
~~~python
comments = pd.read_csv("./data/GBcomments.csv")
~~~
![error](./images/001.png)

Lo solucionamos poniendo `error_bad_lines = False`
~~~python
comments = pd.read_csv("./data/GBcomments.csv", error_bad_lines = False)
~~~
![error](./images/002.png)

<hr>

<a name="schema3"></a>

# 3. Vamos hacer una prueba de análisis de una frase
La puntuación de polaridad es un valor flotante dentro del rango [-1.0, 1.0]. 
La subjetividad es un flotador dentro del rango [0.0, 1.0] donde 0.0 es muy objetivo y 1.0 es muy subjetivo.

~~~python
text = comments.iloc[0].comment_text
~~~
![img](./images/003.png)
En este caso el mensaje es bastante positivo
~~~python
TextBlob(text).sentiment.polarity
~~~
![img](./images/004.png)

<hr>

<a name="schema4"></a>

# 4. Vamos a comprobar si hay nulos y si hay nulos los eliminamos.
Comprobamos la cantidad de elementos que hay en `comments`, como vemos hay bastantes comentarios, ahora vamos a ver si hay nulos y sí los hay los podremos borrar ya que el número de comentarios es muy grande.
~~~python
comments.shape
~~~
![img](./images/006.png)

~~~python
comments.isna().sum()
~~~
![img](./images/005.png)

Lo eliminamos
~~~python
comments.dropna(inplace= True)
~~~

![img](./images/007.png)

<hr>

<a name="schema5"></a>

# 5. Vamos a crear una lista con todas la polaridades de los comentarios y añadirlo al dataset como una columna nueva
~~~python
polarity = []
for comment in comments["comment_text"]:
    polarity.append(TextBlob(comment).sentiment.polarity)

comments["polarity"] = polarity
~~~
![img](./images/008.png)
<hr>

<a name="schema6"></a>

# 6.  Creamos un dataset nuevo que solo contenga los valores de polarity = 1

~~~python
comments_positive = comments[comments["polarity"] == 1]
comments_positive.head()
~~~
![img](./images/009.png)
~~~python
comments_positive.shape
~~~
![img](./images/010.png)

<hr>

<a name="schema7"></a>

# 7. Instalamos WordCloud e importamos la librería

https://amueller.github.io/word_cloud/auto_examples/masked.html?highlight=stopwords
~~~python
conda install -c conda-forge wordcloud
~~~
~~~python
from wordcloud import WordCloud,STOPWORDS
~~~
<hr>

<a name="schema8"></a>

# 8 Creamos la lista de `STOPWORDS` y hacemos que los comentarios es una cadena
~~~python
stopwords = set(STOPWORDS)
total_comments = " ".join(comments_positive["comment_text"])
~~~
![img](./images/011.png)

<hr>

<a name="schema9"></a>

# 9. Creamos el tamaño de imagen
~~~python
wordcloud = WordCloud(width = 1000, height= 500, stopwords= stopwords).generate(total_comments)
~~~

<hr>

<a name="schema10"></a>

# 10. Dibujamos la figura con el texto más positivo
~~~python
plt.figure(figsize=(15,5))
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig("./images/WordCloud.png")
~~~
![worlcloud](./images/WordCloud.png)

<hr>

<a name="schema11"></a>

# 11. Dibujamos la figura con el texto más negativo
~~~python
comments_negative = comments[comments["polarity"] == -1]
total_comments_neg = " ".join(comments_negative["comment_text"])

wordcloud = WordCloud(width = 1000, height= 500, stopwords= stopwords).generate(total_comments_neg)

plt.figure(figsize=(15,5))
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig("./images/WordCloud_neg.png")
~~~
![worlcloud](./images/WordCloud_neg.png)


<hr>

<a name="schema12"></a>

# 12. Cargamos librerías y datos

~~~python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
videos = pd.read_csv("./data/USvideos.csv", error_bad_lines = False)
~~~
![img](./images/012.png)
<hr>

<a name="schema13"></a>

# 13 . Creamos una cadena con todos los tags y le aplicamos una expersión regular
Primero le quitamos cualquier símbolo que no sean letras, 
Segundo le quitamos si tiene más de un espacio
~~~python
tags_complete = " ".join(videos["tags"])
~~~
![img](./images/013.png)
~~~python
tags = re.sub('[^a-zA-Z]',' ', tags_complete)
~~~
![img](./images/014.png)
~~~python
tags = re.sub(' +', ' ', tags)
~~~
![img](./images/015.png)
<hr>

<a name="schema14"></a>

# 14. Dibujamos la figura con tags

~~~
stopwords = set(STOPWORDS)
wordcloud = WordCloud(width = 1000, height= 500, stopwords= stopwords).generate(total_comments_neg)

plt.figure(figsize=(15,5))
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig("./images/WordCloud_neg.png")
~~~
![worlcloud](./images/WordCloud_tags.png)

<hr>

<a name="schema15"></a>

# 15. Dibujamos regresiones con los likes y dislikes

~~~python
sns.regplot(data = videos, x = "views", y = "likes")
plt.title("Regression plot for views & likes")
plt.savefig("./images/regression.png")
~~~
![regression](./images/regression.png)
~~~python
sns.regplot(data = videos, x = "views", y = "dislikes")
plt.title("Regression plot for views & dislikes")
plt.savefig("./images/regression_dislikes.png")
~~~
![regression](./images/regression_dislikes.png)


<hr>

<a name="schema16"></a>

# 16. Vamos a generar una matrix de correlacción
~~~python
sns.heatmap(df_corr.corr(), annot =True)
plt.title("Correlation")
plt.savefig("./images/corr.png")
~~~
![corr](./images/regression_dislikes.png)