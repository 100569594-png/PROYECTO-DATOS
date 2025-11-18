import os

# Definición de la estructura
folders = [
    "data/raw",          # Datos originales (no tocar)
    "data/processed",    # Datos limpios para entrenar
    "notebooks",         # Notebooks para EDA y pruebas
    "src",               # Código fuente modular (.py)
    "results/figures",   # Gráficas generadas
    "results/models"     # Modelos guardados (.pth, .pkl)
]

files = [
    "README.md",            # Memoria del proyecto
    "requirements.txt",     # Librerías necesarias
    "src/__init__.py",      # Para que python reconozca la carpeta como paquete
    "src/preprocessing.py", # Script para limpieza de texto
    "src/models.py",        # Script para definir la red neuronal
    "src/utils.py",         # Funciones auxiliares
    ".gitignore"            # Archivos a ignorar por git
]

# Crear carpetas
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    # Crear un archivo .gitkeep para que git suba la carpeta aunque esté vacía
    with open(os.path.join(folder, ".gitkeep"), "w") as f:
        pass
    print(f"Carpeta creada: {folder}")

# Crear archivos vacíos
for file in files:
    if not os.path.exists(file):
        with open(file, "w") as f:
            pass
        print(f"Archivo creado: {file}")
    else:
        print(f"El archivo ya existe: {file}")

print("\n✅ ¡Estructura del proyecto creada con éxito!")
