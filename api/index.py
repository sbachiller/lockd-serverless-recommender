from http.server import BaseHTTPRequestHandler
from urllib import parse
import csv
import os
import pandas as pd # instalar
import math 
import numpy as np
from pymongo import MongoClient # instalar
import operator
import time

class handler(BaseHTTPRequestHandler):

  def do_GET(self):
    #Tamano del vecindario
    k = 10

    # Sacar vecindario
    def maxN(elements, n):
      el = enumerate(elements)
      cleanedList = [x for x in el if not np.isnan(x[1]) and x[0] != 0]
      sel = sorted(cleanedList, reverse=True, key=operator.itemgetter(1))
      return sel[:n]

    # Calcular r(u,x) dado un usuario y un item
    def getRByUserAndItem(matrix, similitudes, item):
      numerador = 0
      denom = 0

      # Calculo en si
      for i in range(len(similitudes)):
        if not np.isnan(matrix.iloc[item,similitudes[i][0]]):
          numerador += (similitudes[i][1] * matrix.iloc[item,similitudes[i][0]])
          denom += abs(similitudes[i][1])

      # Si ningun vecino ha valorado el item retornar NaN
      if denom != 0:
        return numerador/denom
      else:
        return np.nan

    # Calcular r(u,x) para todos los items dado un usuario
    def getRByUser(matrix,similitudes):
      toReturn = []
      for i, row in matrix.iterrows():
        toReturn.append(getRByUserAndItem(matrix, similitudes, i))
      return toReturn

    # Calcular la matriz completa de r(u,x)
    def getRMatrix(corrMatrix, matrix, vec):
      toReturn = []
      for i, column in enumerate(matrix):
        s = maxN(corrMatrix.iloc[:,i], vec)
        toReturn.append(getRByUser(matrix, s))
      return pd.DataFrame(toReturn)

    def removeItemsByUser(rmat, user, userIndex):
      for i, puntuacion in enumerate(user):
        if not np.isnan(puntuacion):
          rmat.iloc[i,userIndex] = np.nan
      return rmat

    def removeRankedItems(rmat, matrix):
      for i in rmat:
        rmat = removeItemsByUser(rmat, matrix.iloc[:,i], i)
      return rmat

    # Conseguir ranking siempre que prediccion != NaN
    def getRankingForUser(userId, test_rmat):
      cleanedList = [x for x in enumerate(test_rmat.iloc[:,userId]) if not np.isnan(x[1]) and x[0] != 0]
      return sorted(cleanedList, reverse=True, key=operator.itemgetter(1))

    # Calcular todo rankings para todos los usuarios
    def getPredictedMatrix(test_rmat):
      toReturn = []

      for userId in test_rmat:
        rnk = getRankingForUser(userId, test_rmat)
        toReturn.append([x[0] for x in rnk])
      return pd.DataFrame(toReturn).transpose()

    # Sacar array con los indices de un ranking
    def getRankingTitlesByUser(rankingMatrix, userid):
      ranking = rankingMatrix.iloc[:,userid]
      toReturn = []

      for i in ranking:
        if not np.isnan(i):
          toReturn.append(int(i))

      return toReturn


    # DB init
    client = MongoClient(os.environ['MONGO_URI'])
    tic = time.perf_counter()
    db = client.myFirstDatabase
    toc = time.perf_counter()

    # print("CONECTADO!")
    print(f"Connected in {toc - tic:0.4f} seconds")

    # Generate full matrix to calculate corr
    tic = time.perf_counter()
    users = db.users.find().sort([("createdAt", 1)])
    count_rooms = db.rooms.count_documents({})
    full_matrix = []
    toc = time.perf_counter()
    print(f"Got users and rooms in {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    for user in users:
      user_ratings = np.zeros(count_rooms)

      for rating in db.userroomratings.find({'user_id' : user['googleId']}).sort([('room_id', 1)]):
        user_ratings[rating['room_id']] = rating['rate']
      user_ratings = np.where(user_ratings==0, np.NaN, user_ratings) 
      full_matrix.append(user_ratings)

    fullMatrix = pd.DataFrame(full_matrix).T
    fullMatrixCorr = fullMatrix.corr()
    toc = time.perf_counter()
    print(f"Got reviews matrix in {toc - tic:0.4f} seconds")
    # fullMatrixCorr

    # Variables para ejemplo
    nItem = 1
    currUser = 0
    kVecindario = 10

    # Calcular la matriz de r(u,x) completa
    tic = time.perf_counter()
    rmat = getRMatrix(fullMatrixCorr,fullMatrix, 10).transpose()
    rmat.T
    toc = time.perf_counter()
    print(f"Got R(u,x) matrix in {toc - tic:0.4f} seconds")

    # Comparacion de todos los items para usuario = 3
    # pred = rmat.iloc[:,0]
    # trainingItems = fullMatrix.iloc[:,0]
    # frame = {'Item':trainingItems, 'Predicted':pred, 'Real':fullMatrix.iloc[:,0] }
    # df = pd.DataFrame(frame)
    # df.transpose()

    # Eliminar datos de training de la matriz r(u,x)
    tic = time.perf_counter()
    test_rmat = removeRankedItems(rmat, fullMatrix)
    toc = time.perf_counter()
    print(f"Removed training in {toc - tic:0.4f} seconds")

    # Sacar los rankings de todos los usuarios
    tic = time.perf_counter()
    predictedMatrix = getPredictedMatrix(test_rmat)
    toc = time.perf_counter()
    print(f"Predicted in {toc - tic:0.4f} seconds")

    # predictions = getRankingTitlesByUser(predictedMatrix, 0)

    tic = time.perf_counter()
    rooms = [x for x in list(db.rooms.find().sort([('id',1)]))]
    users = [x['googleId'] for x in list(db.users.find().sort([("createdAt", 1)]))]
    toc = time.perf_counter()
    print(f"Got user and room list in {toc - tic:0.4f} seconds")

    # Crear los objetos y subirlos a la coleccion
    tic = time.perf_counter()
    recommendations = []
    db.recommendations.delete_many({}) # Vaciar todas las recomendaciones antes
    for user in predictedMatrix:
      for room in predictedMatrix[user]:
        if not np.isnan(room):
          new_recommendation = {
            'user_id': list(users)[user],
            'room': rooms[int(room)],
            'affinity':round(rmat[user][room]*100/5,2)
          }
          recommendations.append(new_recommendation)
    db.recommendations.insert_many(recommendations)
    toc = time.perf_counter()
    print(f"Database updated in {toc - tic:0.4f} seconds")

    self.send_response(200)
    self.send_header('Content-type','text/plain')
    self.end_headers()

    return