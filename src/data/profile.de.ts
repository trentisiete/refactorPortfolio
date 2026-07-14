// translated: true
// sourceHash: fac4b5bb820e9112
import type { ProfileData, ProfileProject } from './profile.es';

export const profileDe: ProfileData = {
  title: 'Werdegang',
  intro: 'Hier versammle ich meine Berufserfahrung, meine Ausbildung und die Auszeichnungen, die meinen Werdegang begleitet haben.',
  tabsLabel: 'Abschnitte des Werdegangs',
  sections: [
    {
      id: 'experience',
      label: 'Berufserfahrung',
      entries: [
        {
          title: 'Data Scientist Intern',
          organization: 'Naudit HPCN',
          period: '2026',
          logo: { initials: 'NH', src: '/logos/naudit.png', alt: 'Logo von Naudit HPCN' },
        },
        {
          title: 'Präsident und Gründer',
          organization: 'Google Developer Group on Campus UAM',
          period: '2025 - heute',
          logo: { initials: 'GDG', src: '/logos/gdg.webp', alt: 'Logo der Google Developer Group' },
        },
        {
          title: 'Webmaster',
          organization: 'AUTOCENTER - VULCASAN SL',
          period: '2024 - 2025',
          logo: { initials: 'VS', src: '/logos/autocenter.png', alt: 'Logo von Driver Center' },
        },
        {
          title: 'Vorstandsmitglied',
          organization: 'Club für IT-Sicherheit (SEIF UAM)',
          period: '2023 - 2024',
          logo: { initials: 'SEIF', src: '/logos/seif.webp', alt: 'Logo von SEIF UAM' },
        },
        {
          title: 'Selbstständiger Unternehmer',
          organization: 'DripInLocker',
          period: '2021 - 2024',
          logo: { initials: 'DI', src: '/logos/dripinlocker.png', alt: 'Logo von DripInLocker' },
        },
        {
          title: 'Kellner',
          organization: 'Saona',
          period: '2022',
          logo: { initials: 'S', src: '/logos/saona.jpeg', alt: 'Logo von Saona' },
        },
      ],
    },
    {
      id: 'education',
      label: 'Ausbildung',
      entries: [
        {
          title: 'Bachelorstudium in Data Science and Engineering',
          organization: 'Autonome Universität Madrid',
          period: '2022 - 2026',
          logo: { initials: 'UAM', src: '/logos/uam.webp', alt: 'Logo der Autonomen Universität Madrid' },
        },
        {
          title: 'Weiterbildung in Cloud Engineering',
          organization: 'Google Cloud',
          period: '2025 - heute',
          logo: { initials: 'GC', src: '/logos/cloud.png', alt: 'Logo von Google Cloud' },
        },
        {
          title: 'Data Engineering',
          organization: 'Google',
          period: '2025',
          logo: { initials: 'G', src: '/logos/google.webp', alt: 'Logo von Google' },
        },
        {
          title: 'Investorenliga: Stelle deine Portfoliostrategie live auf die Probe',
          organization: 'UAM + IronIA Fintech',
          period: '2025',
          logo: { initials: 'LI', src: '/logos/fuam.png', alt: 'Logo der Stiftung der Autonomen Universität Madrid' },
        },
        {
          title: 'Data Analytics',
          organization: 'Google',
          period: '2024',
          logo: { initials: 'G', src: '/logos/google.webp', alt: 'Logo von Google' },
        },
        {
          title: 'Data Engineering',
          organization: 'Nanjing University',
          period: '2024',
          logo: { initials: 'NJU', src: '/logos/nanjing.png', alt: 'Logo der Nanjing University' },
        },
        {
          title: 'Programm UAM Emprende',
          organization: 'Autonome Universität Madrid',
          period: '2023',
          logo: { initials: 'UAM', src: '/logos/uam_emprende.png', alt: 'Logo von UAM Emprende' },
        },
        {
          title: 'Ausbildung zum Personal Trainer',
          organization: 'Trainologym',
          period: '2021 - 2023',
          logo: { initials: 'T', src: '/logos/trainologym.jpg', alt: 'Logo von Trainologym' },
        },
        {
          title: 'Grundlagen des digitalen Marketings',
          organization: 'Google Digital Garage',
          period: '2021',
          logo: { initials: 'G', src: '/logos/google.webp', alt: 'Logo von Google' },
        },
      ],
    },
    {
      id: 'awards',
      label: 'Auszeichnungen',
      entries: [
        {
          title: 'Stipendium für akademische Exzellenz',
          organization: 'Comunidad de Madrid',
          period: '2024 und 2025',
          logo: { initials: 'CM', src: '/logos/ccaa_mad.webp', alt: 'Logo der Comunidad de Madrid' },
        },
        {
          title: '1. Platz - Investorenliga der UAM',
          organization: 'Autonome Universität Madrid + IronIA Fintech',
          period: '2025',
          logo: { initials: 'LI', src: '/logos/fuam.png', alt: 'Logo der Stiftung der Autonomen Universität Madrid' },
        },
        {
          title: '2. Platz - Hackathon für Cloud-Agentensysteme',
          organization: 'Google Cloud + Diverger',
          period: '2025',
          logo: { initials: 'GC', src: '/logos/cloud.png', alt: 'Logo von Google Cloud' },
        },
        {
          title: 'Akademische Anerkennung für herausragende Leistungen',
          organization: 'Colegio San José de Begoña',
          period: '2020 und 2021',
          logo: { initials: 'CSJ', src: '/logos/colegio.png', alt: 'Wappen des Colegio San José de Begoña' },
        },
      ],
    },
  ],
};

export const profileProjectsDe: ProfileProject[] = [
  {
    title: 'Ersatzmodelle zur Suche nach optimalen Eingaben',
    excerpt: 'Gaußsche Prozesse, Bayes-Optimierung und Expected Improvement zur Suche nach guten Konfigurationen bei begrenztem Auswertungsbudget.',
    year: '2026',
    github: 'https://github.com/trentisiete/surrogate_models',
    articleHref: '/de/articles/surrogate_models/',
  },
  {
    title: 'RoadGuard',
    excerpt: 'Zuverlässige Erkennung von Straßenschäden: länderübergreifender Transfer, Konfidenzkalibrierung, Abstentionsstrategie und erklärbare Priorisierung der Instandhaltung.',
    year: '2026',
    github: 'https://github.com/trentisiete/RoadGuard',
  },
  {
    title: 'BGG Review Intelligence',
    excerpt: 'Von BoardGameGeek-Rezensionen zu strukturiertem Meinungswissen: linguistische Vorverarbeitung, Sentiment-Klassifikation und aspektbasierte Analyse.',
    year: '2025',
    articleHref: '/de/articles/bgg_review_intelligence/',
  },
  {
    title: 'Bilderzeugung mit Diffusionsmodellen',
    excerpt: 'Implementierung des SDE-Frameworks von Song et al.: Training per Denoising Score Matching, Sampler, bedingte Generierung und Imputation auf CIFAR-10.',
    year: '2025',
    github: 'https://github.com/trentisiete/DiffusionImaGen',
    articleHref: '/de/articles/diffusion_models/',
  },
];
