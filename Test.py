import Actividades as A
import Utils as U
import numpy as np
import matplotlib.pyplot as plt
# Carga la imagen para Actividad 1
img = A.ImagenTermo.from_file('Imagen_0.xlsx')  # Carga archivo específico
img.plot_divisions()  # Grafica divisiones
img.plot_selected_subquadrants()  # Grafica subcuadrantes seleccionados
img.plot_quadrants_with_histograms()  # Grafica cuadrantes con histogramas
# Para Actividad 2 con múltiples pacientes
patient1_files = ['Imagen_0.xlsx', 'Imagen_1000.xlsx', 'Imagen_2000.xlsx', 'Imagen_3000.xlsx', 'Imagen_4000.xlsx', 'Imagen_5000.xlsx']  # Archivos paciente 1
patient2_files = ['Imagen2_0.xlsx', 'Imagen2_1000.xlsx', 'Imagen2_2000.xlsx', 'Imagen2_3000.xlsx', 'Imagen2_4000.xlsx', 'Imagen2_5000.xlsx']  # Archivos paciente 2
patient_files_list = [patient1_files, patient2_files]  # Lista de pacientes
U.generate_stats_tables(patient_files_list)  # Genera tablas
# Para Actividad 3 con Imagen_0.xlsx
img = A.ImagenTermo.from_file('Imagen_0.xlsx')  # Carga archivo
img.plot_comparative_subquad_histograms()  # Grafica histogramas comparativos
img.plot_delta_heatmap()  # Grafica mapa de calor de deltas
# Gráfico de evolución para paciente 1
doses = []  # Lista para dosis
avg_deltas = []  # Lista para deltas promedio
for file in patient1_files:  # Para cada archivo del paciente 1
    img = A.ImagenTermo.from_file(file)  # Carga imagen
    delta_sum = 0  # Suma de deltas
    count = 0  # Contador
    for rr in range(2):  # Para filas 2 y 3
        r = rr + 1  # Ajusta índice
        for s in range(3):  # Para subcuadrantes 1,2,3
            temps_non = img.get_subquad_temps(r, s+1, 'no_irradiado')  # Temperaturas no irradiadas
            temps_irr = img.get_subquad_temps(r, s+1, 'irradiado')  # Temperaturas irradiadas
            mean_non = np.mean(temps_non) if len(temps_non) > 0 else 0  # Media no irradiada
            mean_irr = np.mean(temps_irr) if len(temps_irr) > 0 else 0  # Media irradiada
            delta_sum += mean_irr - mean_non  # Suma delta
            count += 1  # Incrementa contador
    avg_delta = delta_sum / count if count > 0 else 0  # Calcula promedio
    doses.append(img.dose)  # Añade dosis
    avg_deltas.append(avg_delta)  # Añade delta promedio
plt.plot(doses, avg_deltas, marker='o', color='red')  # Grafica línea con marcadores
plt.title('Evolución del delta de temperatura con respecto al tiempo de las dosis')  # Título
plt.xlabel('Dosis (cGy)')  # Etiqueta x
plt.ylabel('Delta promedio (°C)')  # Etiqueta y
plt.grid(True)  # Agrega rejilla
plt.show()  # Muestra figura