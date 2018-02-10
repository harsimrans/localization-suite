
data<-read.csv("data12.csv")
mymap<-leaflet(data=data)
mymap<-addTiles(mymap)
mymap<-addCircleMarkers(mymap, ~LONG,~LAT,radius=1,color='blue')
mymap<-addCircleMarkers(mymap,-112.074635, 33.444065, radius=10,color='red')
mymap