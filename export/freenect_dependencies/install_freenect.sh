#!/bin/bash

# Ruta donde se instalarán las bibliotecas
INSTALL_DIR=/usr/local/lib

# Copiar las bibliotecas
echo "Instalando dependencias de freenect..."
cp libfreenect_sync.so.0 $INSTALL_DIR/
cp libfreenect.so.0 $INSTALL_DIR/
cp libpython3.10.so.1.0 $INSTALL_DIR/

# Actualizar caché de bibliotecas dinámicas
ldconfig

echo "Instalación completa."
