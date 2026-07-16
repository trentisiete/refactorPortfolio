import { createProjectGallery } from './projectMedia';

export type ProfileSectionId = 'experience' | 'education' | 'awards';

export interface ProfileLink {
  label: string;
  href: string;
  external?: boolean;
}

export interface ProfileLogo {
  initials: string;
  src?: string;
  alt?: string;
}

export interface ProfileEntry {
  title: string;
  organization: string;
  period?: string;
  location?: string;
  logo: ProfileLogo;
  details?: string[];
  links?: ProfileLink[];
}

export interface ProfileSection {
  id: ProfileSectionId;
  label: string;
  entries: ProfileEntry[];
}

export interface ProfileProject {
  id: string;
  title: string;
  excerpt?: string;
  year?: string;
  github?: string;
  articleHref?: string;
  gallery?: ProfileProjectImage[];
  notes?: ProfileProjectNote[];
}

export interface ProfileProjectImage {
  src: string;
  alt: string;
  caption?: string;
}

export type ProfileProjectNoteTone = 'yellow' | 'blue' | 'green' | 'violet' | 'neutral';

export interface ProfileProjectNote {
  text: string;
  tone?: ProfileProjectNoteTone;
}

export interface ProfileData {
  title: string;
  intro: string;
  tabsLabel: string;
  sections: ProfileSection[];
}

export const profileEs: ProfileData = {
  title: 'Trayectoria',
  intro: 'Aquí reúno mi experiencia profesional, mi formación y los reconocimientos que han acompañado mi trayectoria.',
  tabsLabel: 'Secciones de trayectoria',
  sections: [
    {
      id: 'experience',
      label: 'Experiencia',
      entries: [
        {
          title: 'Data Scientist Intern',
          organization: 'Naudit HPCN',
          period: '2026',
          logo: { initials: 'NH', src: '/logos/naudit.png', alt: 'Logo de Naudit HPCN' },
        },
        {
          title: 'Presidente y fundador',
          organization: 'Google Developer Group on Campus UAM',
          period: '2025 - actualidad',
          logo: { initials: 'GDG', src: '/logos/gdg.webp', alt: 'Logo de Google Developer Group' },
        },
        {
          title: 'Webmaster',
          organization: 'AUTOCENTER - VULCASAN SL',
          period: '2024 - 2025',
          logo: { initials: 'VS', src: '/logos/autocenter.png', alt: 'Logo de Driver Center' },
        },
        {
          title: 'Miembro de la junta directiva',
          organization: 'Club de Seguridad Informática (SEIF UAM)',
          period: '2023 - 2024',
          logo: { initials: 'SEIF', src: '/logos/seif.webp', alt: 'Logo de SEIF UAM' },
        },
        {
          title: 'Emprendedor independiente',
          organization: 'DripInLocker',
          period: '2021 - 2024',
          logo: { initials: 'DI', src: '/logos/dripinlocker.png', alt: 'Logo de DripInLocker' },
        },
        {
          title: 'Camarero',
          organization: 'Saona',
          period: '2022',
          logo: { initials: 'S', src: '/logos/saona.jpeg', alt: 'Logo de Saona' },
        },
      ],
    },
    {
      id: 'education',
      label: 'Educación',
      entries: [
        {
          title: 'Grado en Ciencia e Ingeniería de Datos',
          organization: 'Universidad Autónoma de Madrid',
          period: '2022 - 2026',
          logo: { initials: 'UAM', src: '/logos/uam.webp', alt: 'Logo de la Universidad Autónoma de Madrid' },
        },
        {
          title: 'Formación en Cloud Engineering',
          organization: 'Google Cloud',
          period: '2025 - actualidad',
          logo: { initials: 'GC', src: '/logos/cloud.png', alt: 'Logo de Google Cloud' },
        },
        {
          title: 'Data Engineering',
          organization: 'Google',
          period: '2025',
          logo: { initials: 'G', src: '/logos/google.webp', alt: 'Logo de Google' },
        },
        {
          title: 'Liga de Inversores: Reta tu Estrategia de Carteras en Vivo',
          organization: 'UAM + IronIA Fintech',
          period: '2025',
          logo: { initials: 'LI', src: '/logos/fuam.png', alt: 'Logo de la Fundación Universidad Autónoma de Madrid' },
        },
        {
          title: 'Data Analytics',
          organization: 'Google',
          period: '2024',
          logo: { initials: 'G', src: '/logos/google.webp', alt: 'Logo de Google' },
        },
        {
          title: 'Data Engineering',
          organization: 'Nanjing University',
          period: '2024',
          logo: { initials: 'NJU', src: '/logos/nanjing.png', alt: 'Logo de Nanjing University' },
        },
        {
          title: 'Programa UAM Emprende',
          organization: 'Universidad Autónoma de Madrid',
          period: '2023',
          logo: { initials: 'UAM', src: '/logos/uam_emprende.png', alt: 'Logo de UAM Emprende' },
        },
        {
          title: 'Formación en Entrenamiento Personal',
          organization: 'Trainologym',
          period: '2021 - 2023',
          logo: { initials: 'T', src: '/logos/trainologym.jpg', alt: 'Logo de Trainologym' },
        },
        {
          title: 'Fundamentos de Marketing Digital',
          organization: 'Google Digital Garage',
          period: '2021',
          logo: { initials: 'G', src: '/logos/google.webp', alt: 'Logo de Google' },
        },
      ],
    },
    {
      id: 'awards',
      label: 'Premios',
      entries: [
        {
          title: 'Beca de Excelencia Académica',
          organization: 'Comunidad de Madrid',
          period: '2024 y 2025',
          logo: { initials: 'CM', src: '/logos/ccaa_mad.webp', alt: 'Logo de la Comunidad de Madrid' },
        },
        {
          title: '1.er puesto - Liga de Inversores UAM',
          organization: 'Universidad Autónoma de Madrid + IronIA Fintech',
          period: '2025',
          logo: { initials: 'LI', src: '/logos/fuam.png', alt: 'Logo de la Fundación Universidad Autónoma de Madrid' },
        },
        {
          title: '2.º puesto - Hackathon de Sistemas de Agentes Cloud',
          organization: 'Google Cloud + Diverger',
          period: '2025',
          logo: { initials: 'GC', src: '/logos/cloud.png', alt: 'Logo de Google Cloud' },
        },
        {
          title: 'Reconocimiento académico por alto rendimiento',
          organization: 'Colegio San José de Begoña',
          period: '2020 y 2021',
          logo: { initials: 'CSJ', src: '/logos/colegio.png', alt: 'Escudo del Colegio San José de Begoña' },
        },
      ],
    },
  ],
};

export const profileProjectsEs: ProfileProject[] = [
  {
    id: 'surrogate-models',
    title: 'Modelos sustitutos para la búsqueda de entradas óptimas',
    excerpt: 'Procesos gaussianos, optimización bayesiana y Expected Improvement para buscar buenas configuraciones con un presupuesto limitado de evaluaciones.',
    year: '2026',
    github: 'https://github.com/trentisiete/surrogate_models',
    articleHref: '/es/articles/surrogate_models/',
    notes: [
      { text: 'Trabajo de Fin de Grado', tone: 'yellow' },
    ],
    gallery: createProjectGallery('surrogate-models', 'Resultado visual de modelos sustitutos', {
      'app_forrester_ei_decision_synthesis_step5.png': 'Síntesis de una decisión de Expected Improvement en Forrester: posterior del GP, criterio de adquisición y efecto sobre el incumbent.',
      'app_forrester_gp_evolution_no_noise.png': 'Cómo evoluciona el proceso gaussiano al incorporar nuevas evaluaciones.',
      'app_real_best_gp_parity_by_target.png': 'Observado frente a predicho en validación LODO para el mejor GP de cada objetivo del caso real.',
      'app_real_dataset_targets_by_diet.png': 'Valores medios observados por dieta para los tres objetivos: la comprobación descriptiva previa al modelado.',
      'app_real_observed_equal_weight_ranking.png': 'Ranking exploratorio de las dietas observadas con una puntuación de pesos iguales.',
      'evolution_incumbent_best_model_by_benchmark.png': 'Evolución de la mejora del mínimo encontrado durante el infill para el mejor modelo de cada benchmark.',
      'evolution_relative_mae_best_model_by_benchmark.png': 'Evolución del MAE relativo durante el infill para el mejor modelo de cada benchmark.',
      'factor_effects_paired_for_tfg.png': 'Efectos emparejados de las condiciones experimentales: sampler, ARD y ruido sobre el incumbent y el MAE.',
      'fig_01_lodo_generalization_vs_dummy.png': 'Rendimiento relativo frente al baseline Dummy en validación LODO; por debajo de 1, el modelo generaliza mejor.',
      'fig_02_error_by_left_out_diet.png': 'Dificultad de generalización por dieta retenida: error normalizado y MAE en las unidades originales.',
      'fig_04_kernel_family_mae.png': 'Comparación de familias de kernel por MAE macro en el caso real de Hermetia.',
      'fig_05_uncertainty_vs_error.png': 'Diagnóstico de la relación entre incertidumbre y error predictivo.',
      'fig_06_prediction_intervals_by_diet.png': 'Intervalos de predicción del GP por dieta retenida en validación LODO.',
      'fig_08_ei_candidate_landscape.png': 'Priorización de nuevos candidatos mediante Expected Improvement.',
      'fig_ei_demo.png': 'El ciclo de infill con Expected Improvement en Forrester: dos pasos de posterior y adquisición.',
      'fig_gp_prior_posterior.png': 'Muestras del prior de un proceso gaussiano con kernel RBF y posterior tras dos observaciones.',
      'fig_kernel_matern52.png': 'Prior y posterior con kernel Matérn 5/2: trayectorias más rugosas que con RBF.',
      'final_relative_improvement_mae_by_benchmark.png': 'Mejora predictiva final del MAE por benchmark y kernel, con intervalos de confianza.',
      'final_relative_incumbent_improvement_by_benchmark.png': 'Mejora final del mínimo encontrado por benchmark y kernel al terminar el infill.',
      'mae_vs_probabilistic_diagnostics_by_step.png': 'Mejora del error puntual frente a la calidad probabilística durante el infill; el color mide la calibración de la cobertura.',
    }),
  },
  {
    id: 'roadguard',
    title: 'RoadGuard',
    excerpt: 'Detección fiable de daños viales: transferencia entre países, calibración de confianza, política de abstención y priorización explicable del mantenimiento.',
    year: '2026',
    github: 'https://github.com/trentisiete/RoadGuard',
    articleHref: '/es/articles/roadguard/',
    gallery: createProjectGallery('roadguard', 'Resultado visual de RoadGuard', {
      'bootstrap_f1_ci.png': 'F1 por país con intervalos bootstrap del 95%: la calibración no mejora F1 de forma uniforme.',
      'calibration_reliability.png': 'Diagrama de fiabilidad antes y después de calibrar: el ECE cae de 0.089 a 0.029.',
      'confidence_shift_by_correctness.png': 'La calibración isotónica separa aciertos de errores: los fallos quedan en confianzas bajas, donde el sistema puede abstenerse.',
      'example_1_China_MotorBike.jpg': 'Ejemplo de inferencia sobre daños viales en el dominio de China.',
      'example_3_China_MotorBike.jpg': 'Detección de grieta transversal en el dominio de China.',
      'example_4_United_States.jpg': 'Transferencia del sistema al dominio de Estados Unidos.',
      'example_5_United_States.jpg': 'Grietas alligator y longitudinal detectadas en Estados Unidos.',
      'per_class_ap_by_country.png': 'AP50 por clase y país con la ontología corregida: Pothole fuerte en China y más débil al transferir.',
      'policy_comparison.png': 'Precisión, recall y F1 por dominio: política base frente a política calibrada.',
      'risk_coverage_curve.png': 'Frontera riesgo-cobertura de la predicción selectiva con el punto de operación de cada política.',
      'stress_robustness.png': 'Prueba de estrés ante oscuridad y desenfoque.',
    }),
  },
  {
    id: 'bgg-review-intelligence',
    title: 'BGG Review Intelligence',
    excerpt: 'De reseñas de BoardGameGeek a conocimiento estructurado de opinión: preprocesado lingüístico, clasificación de sentimiento y análisis basado en aspectos.',
    year: '2025',
    github: 'https://github.com/trentisiete/bgg-review-intelligence',
    articleHref: '/es/articles/bgg_review_intelligence/',
    gallery: createProjectGallery('bgg-review-intelligence', 'Resultado visual de BGG Review Intelligence'),
  },
  {
    id: 'diffusion-models',
    title: 'Generación de imágenes con modelos de difusión',
    excerpt: 'Implementación del marco de SDEs de Song et al.: entrenamiento por denoising score matching, samplers, generación condicional e imputación sobre CIFAR-10.',
    year: '2025',
    github: 'https://github.com/trentisiete/DiffusionImaGen',
    articleHref: '/es/articles/diffusion_models/',
    gallery: createProjectGallery('diffusion-models', 'Resultado visual de modelos de difusión', {
      'bpd.jpeg': 'Distribución de BPD (bits por dimensión) entre configuraciones SDE.',
      'classifier_architecture_image.png': 'Arquitectura del clasificador dependiente del tiempo que guía la generación condicional.',
      'comparativa_samplers_vp_lineal.png': 'Comparación de resultados entre cuatro estrategias de muestreo.',
      'evolucion_muestras_vp_lineal.png': 'Del ruido inicial a la muestra final mediante VP-SDE.',
      'fid.jpeg': 'Distribución de FID entre las distintas configuraciones SDE.',
      'final_conditional_comparison.png': 'Comparación de generación condicional entre configuraciones.',
      'image_dcfe22.png': 'Trayectoria de denoising paso a paso: del ruido puro a la imagen tras mil pasos.',
      'imputacion_cifar_mnist_3etapas.png': 'Reconstrucción de regiones enmascaradas mediante difusión.',
      'is_media.jpeg': 'Inception Score medio por configuración SDE.',
      'is_std.jpeg': 'Desviación estándar interna del Inception Score por configuración.',
      'loss_average_barplot_all_sdes.png': 'Loss promedio del entrenamiento inicial por configuración SDE.',
      'loss_curves_all_sdes.png': 'Curvas de loss durante las primeras 50 épocas para cada SDE.',
      'loss_final_barplot_all_sdes.png': 'Loss final tras 50 épocas por configuración SDE.',
      'loss_violin_plot_all_sdes.png': 'Distribución del loss por configuración durante el entrenamiento inicial.',
      'mi_diagrama_gantt.jpeg': 'Planificación temporal del proyecto: primera fase del diagrama de Gantt.',
      'mi_diagrama_gantt2.jpeg': 'Planificación temporal del proyecto: segunda fase del diagrama de Gantt.',
    }),
  },
];
