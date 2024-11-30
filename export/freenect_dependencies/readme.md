# Freenect Dependencies Export

Este directorio contiene las dependencias necesarias para utilizar la biblioteca `freenect` en sistemas ARM de 64 bits. La instalación incluye todas las bibliotecas requeridas y un script de instalación para facilitar el proceso.

---

## Contenido de la Carpeta

- **`freenect.so`**: Archivo principal de la biblioteca `freenect` compilado para Python 3.10.
- **`libfreenect_sync.so.0`**: Biblioteca de sincronización de `libfreenect`.
- **`libfreenect.so.0`**: Biblioteca principal de `libfreenect`.
- **`libpython3.10.so.1.0`**: Biblioteca dinámica de Python 3.10.
- **`install_freenect.sh`**: Script de instalación para copiar las bibliotecas al sistema y configurarlas correctamente.

---

## Requisitos Previos

Antes de comenzar, asegúrate de que el sistema de destino cumpla con los siguientes requisitos:

1. **Sistema operativo**: ARM de 64 bits (aarch64).
2. **Python 3.10 instalado**: El sistema debe tener Python 3.10 instalado.
3. **Privilegios de administrador**: Necesitarás acceso de superusuario (`sudo`) para instalar las bibliotecas.

---

## Instrucciones de Instalación

1. **Transferir los archivos al sistema de destino**  
   Copia la carpeta completa al sistema donde deseas instalar las dependencias. Usa `scp` o un dispositivo USB. Por ejemplo, si estás usando `scp`:
   ```bash
   scp -r export/freenect_dependencies user@destination:/path/to/destination
   ```

2. **Dar permisos de ejecución al script**  
   En el sistema de destino, navega hasta la carpeta donde copiaste los archivos y haz ejecutable el script de instalación:
   ```bash
   chmod +x install_freenect.sh
   ```

3. **Ejecutar el script de instalación**  
   Ejecuta el script para instalar las dependencias:
   ```bash
   sudo ./install_freenect.sh
   ```
   Este script hará lo siguiente:
   - Copiará las bibliotecas (`libfreenect_sync.so.0`, `libfreenect.so.0`, `libpython3.10.so.1.0`) a `/usr/local/lib`.
   - Actualizará la caché de bibliotecas dinámicas con `ldconfig`.

4. **Mover el archivo `freenect.so`**  
   Copia el archivo `freenect.so` al directorio de `site-packages` de Python 3.10:
   ```bash
   sudo cp freenect.so /usr/lib/python3.10/site-packages/
   ```

---

## Verificación de la Instalación

1. Abre una terminal y ejecuta Python 3.10:
   ```bash
   python3.10
   ```

2. Importa la biblioteca `freenect` para verificar que está instalada correctamente:
   ```python
   import freenect
   print(freenect)
   ```

   Si no aparece ningún error, la instalación fue exitosa.

---

## Uso de la Biblioteca

Una vez instalada, puedes utilizar la biblioteca `freenect` en tus scripts de Python. Por ejemplo:

```python
import freenect

# Conectar al Kinect y capturar datos
depth, timestamp = freenect.sync_get_depth()
print("Profundidad:", depth)
print("Timestamp:", timestamp)
```

---

## Solución de Problemas

1. **Error de librería no encontrada (`libfreenect.so.0` o similar)**  
   Verifica que las bibliotecas se instalaron correctamente en `/usr/local/lib` y que ejecutaste `ldconfig`:
   ```bash
   sudo ldconfig
   ```

2. **Error al importar `freenect`**  
   Asegúrate de que el archivo `freenect.so` esté en el directorio de `site-packages` de Python 3.10.

3. **Python incorrecto**  
   Verifica que estás usando Python 3.10:
   ```bash
   python3.10 --version
   ```

---

## Créditos

- **Autor**: Configuración personalizada para ARM de 64 bits.
- **Dependencias**: Basado en `libfreenect` y las bibliotecas asociadas.
```

### **Cómo usar este README**
- Copia este archivo como `README.md` en el directorio `freenect_dependencies`.
- Transfiérelo junto con la carpeta al sistema de destino.
- Los pasos están claramente detallados para cualquier usuario que desee instalar y usar las dependencias.