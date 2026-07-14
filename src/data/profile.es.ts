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
  title: string;
  excerpt?: string;
  year?: string;
  github?: string;
  articleHref?: string;
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
    title: 'Modelos sustitutos para la búsqueda de entradas óptimas',
    excerpt: 'Procesos gaussianos, optimización bayesiana y Expected Improvement para buscar buenas configuraciones con un presupuesto limitado de evaluaciones.',
    year: '2026',
    github: 'https://github.com/trentisiete/surrogate_models',
    articleHref: '/es/articles/surrogate_models/',
  },
  {
    title: 'RoadGuard',
    excerpt: 'Detección fiable de daños viales: transferencia entre países, calibración de confianza, política de abstención y priorización explicable del mantenimiento.',
    year: '2026',
    github: 'https://github.com/trentisiete/RoadGuard',
  },
  {
    title: 'BGG Review Intelligence',
    excerpt: 'De reseñas de BoardGameGeek a conocimiento estructurado de opinión: preprocesado lingüístico, clasificación de sentimiento y análisis basado en aspectos.',
    year: '2025',
    articleHref: '/es/articles/bgg_review_intelligence/',
  },
  {
    title: 'Generación de imágenes con modelos de difusión',
    excerpt: 'Implementación del marco de SDEs de Song et al.: entrenamiento por denoising score matching, samplers, generación condicional e imputación sobre CIFAR-10.',
    year: '2025',
    github: 'https://github.com/trentisiete/DiffusionImaGen',
    articleHref: '/es/articles/diffusion_models/',
  },
];
