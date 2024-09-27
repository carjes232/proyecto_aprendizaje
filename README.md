# Desarrollo de un Modelo de Detección de Objetos Rápido y Eficiente para Raspberry Pi

## Descripción
Este proyecto desarrolla y optimiza modelos de detección de objetos para su ejecución en tiempo real en una Raspberry Pi. Se evaluaron diferentes arquitecturas ligeras como YOLOv5n, YOLOv8n y NanoDet-Plus-m_320, implementando técnicas de cuantización y optimización para mejorar el rendimiento.

## Estructura del Repositorio

- `PC/`: Contiene los scripts de entrenamiento, los modelos y los datasets utilizados para entrenar y optimizar los modelos.
- `RaspberryPi/`: Incluye los modelos optimizados, la aplicación de inferencia y la integración con Kinect para la detección de objetos en tiempo real.
- `docs/`: Documentación adicional, incluyendo el diagrama de Gantt y el escudo de la universidad.
- `README.md`: Descripción general del proyecto y guía de uso.
- `requirements.txt`: Dependencias necesarias para ejecutar los scripts.

## Instalación

### Requisitos
- Python 3.8+
- Git
- Raspberry Pi con Raspberry Pi OS
- Kinect

### Clonar el Repositorio
```bash
git clone https://github.com/tu-usuario/proyecto_aprendizaje.git
cd proyecto_aprendizaje
