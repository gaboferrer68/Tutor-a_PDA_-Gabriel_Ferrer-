import Actividades as A
import pandas as pd
# Función para computar datos de un paciente
def compute_patient_data(files, patient_id=1):
    """
    Computa los datos para un solo paciente desde sus archivos.
    Retorna un DataFrame con las estadísticas.
    """
    data = []  # Lista para recopilar datos
    for file in files:  # Para cada archivo
        img = A.ImagenTermo.from_file(file)  # Carga la imagen
        dose = img.dose  # Obtiene la dosis
        stats_irr = img.get_quadrant_stats(29, 'irradiado')  # Estadísticas irradiadas
        stats_non = img.get_quadrant_stats(29, 'no_irradiado')  # Estadísticas no irradiadas
        # Añade diccionario con datos
        data.append({
            'Paciente': patient_id,
            'Dosis': dose,
            'Temp media (irr)': stats_irr['mean'],
            'Var (irr)': stats_irr['var'],
            'std (irr)': stats_irr['std'],
            'max (irr)': stats_irr['max'],
            'min (irr)': stats_irr['min'],
            'Temp media (non)': stats_non['mean'],
            'Var (non)': stats_non['var'],
            'std (non)': stats_non['std'],
            'max (non)': stats_non['max'],
            'min (non)': stats_non['min']
        })
    # Retorna DataFrame ordenado por dosis
    return pd.DataFrame(data).sort_values('Dosis')

# Función para generar tablas de estadísticas
def generate_stats_tables(patient_files_list):
    """
    Genera tablas A, B, C y D para todos los pacientes.
    patient_files_list: Lista de listas, cada lista interna es archivos para un paciente.
    """
    all_df = pd.DataFrame()  # DataFrame vacío para todos los datos
    for patient_id, files in enumerate(patient_files_list, start=1):  # Para cada paciente
        patient_df = compute_patient_data(files, patient_id)  # Computa datos del paciente
        all_df = pd.concat([all_df, patient_df], ignore_index=True)  # Concatena
    # Tabla A: Mama irradiada (todos los pacientes)
    table_a = all_df[['Paciente', 'Dosis', 'Temp media (irr)', 'Var (irr)', 'std (irr)', 'max (irr)', 'min (irr)']]
    table_a.columns = ['Paciente', 'Dosis', 'Temp media', 'Var', 'std', 'max', 'min']  # Renombra columnas
    print("Tabla A - Mama irradiada:")  # Imprime título
    print(table_a.to_string(index=False))  # Imprime tabla sin índice
    # Tabla B: Mama no irradiada (todos los pacientes)
    table_b = all_df[['Paciente', 'Dosis', 'Temp media (non)', 'Var (non)', 'std (non)', 'max (non)', 'min (non)']]
    table_b.columns = ['Paciente', 'Dosis', 'Temp media', 'Var', 'std', 'max', 'min']  # Renombra columnas
    print("Tabla B - Mama no irradiada:")  # Imprime título
    print(table_b.to_string(index=False))  # Imprime tabla sin índice
    # Tabla C: Combinada (todos los pacientes)
    print("Tabla C - Estadísticas totales:")  # Imprime título
    print(all_df.to_string(index=False))  # Imprime tabla sin índice
    # Tabla D: Diferencias (delta) de temperaturas medias (irr - non), pivote para pacientes vs dosis
    deltas = all_df['Temp media (irr)'] - all_df['Temp media (non)']  # Calcula deltas
    all_df['Delta'] = deltas  # Añade columna Delta
    table_d = all_df.pivot(index='Paciente', columns='Dosis', values='Delta')  # Pivotea la tabla
    print("Tabla D - Diferencia (delta) de temperaturas medias:")  # Imprime título
    print(table_d.to_string())  # Imprime tabla