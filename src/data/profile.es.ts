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
      'app_forrester_gp_evolution_no_noise.png': 'Cómo evoluciona el proceso gaussiano al incorporar nuevas evaluaciones.',
      'fig_05_uncertainty_vs_error.png': 'Diagnóstico de la relación entre incertidumbre y error predictivo.',
      'fig_08_ei_candidate_landscape.png': 'Priorización de nuevos candidatos mediante Expected Improvement.',
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
      'example_1_China_MotorBike.jpg': 'Ejemplo de inferencia sobre daños viales en el dominio de China.',
      'example_4_United_States.jpg': 'Transferencia del sistema al dominio de Estados Unidos.',
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
      'evolucion_muestras_vp_lineal.png': 'Del ruido inicial a la muestra final mediante VP-SDE.',
      'comparativa_samplers_vp_lineal.png': 'Comparación de resultados entre cuatro estrategias de muestreo.',
      'imputacion_cifar_mnist_3etapas.png': 'Reconstrucción de regiones enmascaradas mediante difusión.',
    }),
  },
];
