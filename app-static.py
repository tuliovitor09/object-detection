from ultralytics import YOLO

# exemplo com imagem estática
# carregar modelo
model = YOLO("yolo26n.pt")

# detectar objetos
results = model("images/test-2.jpg")

# mostrar resultados
results[0].show()

# imprimir objetos encontrados
for box in results[0].boxes:
    classe = int(box.cls[0])
    confidence = float(box.conf[0])

    print(model.names[classe], confidence)
