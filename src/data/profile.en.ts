// translated: true
// sourceHash: aa766361b055720d
import type { ProfileData, ProfileProject } from './profile.es';

export const profileEn: ProfileData = {
  title: 'Journey',
  intro: 'Here I bring together my professional experience, my education and the recognitions that have accompanied my journey.',
  tabsLabel: 'Journey sections',
  sections: [
    {
      id: 'experience',
      label: 'Experience',
      entries: [
        {
          title: 'Data Scientist Intern',
          organization: 'Naudit HPCN',
          period: '2026',
          logo: { initials: 'NH', src: '/logos/naudit.png', alt: 'Naudit HPCN logo' },
        },
        {
          title: 'President & Founder',
          organization: 'Google Developer Group on Campus UAM',
          period: '2025 - present',
          logo: { initials: 'GDG', src: '/logos/gdg.webp', alt: 'Google Developer Group logo' },
        },
        {
          title: 'Webmaster',
          organization: 'AUTOCENTER - VULCASAN SL',
          period: '2024 - 2025',
          logo: { initials: 'VS', src: '/logos/autocenter.png', alt: 'Driver Center logo' },
        },
        {
          title: 'Board Member',
          organization: 'Computer Security Club (SEIF UAM)',
          period: '2023 - 2024',
          logo: { initials: 'SEIF', src: '/logos/seif.webp', alt: 'SEIF UAM logo' },
        },
        {
          title: 'Independent Entrepreneur',
          organization: 'DripInLocker',
          period: '2021 - 2024',
          logo: { initials: 'DI', src: '/logos/dripinlocker.png', alt: 'DripInLocker logo' },
        },
        {
          title: 'Waiter',
          organization: 'Saona',
          period: '2022',
          logo: { initials: 'S', src: '/logos/saona.jpeg', alt: 'Saona logo' },
        },
      ],
    },
    {
      id: 'education',
      label: 'Education',
      entries: [
        {
          title: "Bachelor's Degree in Data Science and Engineering",
          organization: 'Autonomous University of Madrid',
          period: '2022 - 2026',
          logo: { initials: 'UAM', src: '/logos/uam.webp', alt: 'Autonomous University of Madrid logo' },
        },
        {
          title: 'Cloud Engineering Training',
          organization: 'Google Cloud',
          period: '2025 - present',
          logo: { initials: 'GC', src: '/logos/cloud.png', alt: 'Google Cloud logo' },
        },
        {
          title: 'Data Engineering',
          organization: 'Google',
          period: '2025',
          logo: { initials: 'G', src: '/logos/google.webp', alt: 'Google logo' },
        },
        {
          title: 'Investors League: Challenge Your Portfolio Strategy Live',
          organization: 'UAM + IronIA Fintech',
          period: '2025',
          logo: { initials: 'LI', src: '/logos/fuam.png', alt: 'Autonomous University of Madrid Foundation logo' },
        },
        {
          title: 'Data Analytics',
          organization: 'Google',
          period: '2024',
          logo: { initials: 'G', src: '/logos/google.webp', alt: 'Google logo' },
        },
        {
          title: 'Data Engineering',
          organization: 'Nanjing University',
          period: '2024',
          logo: { initials: 'NJU', src: '/logos/nanjing.png', alt: 'Nanjing University logo' },
        },
        {
          title: 'UAM Emprende Program',
          organization: 'Autonomous University of Madrid',
          period: '2023',
          logo: { initials: 'UAM', src: '/logos/uam_emprende.png', alt: 'UAM Emprende logo' },
        },
        {
          title: 'Personal Training Education',
          organization: 'Trainologym',
          period: '2021 - 2023',
          logo: { initials: 'T', src: '/logos/trainologym.jpg', alt: 'Trainologym logo' },
        },
        {
          title: 'Fundamentals of Digital Marketing',
          organization: 'Google Digital Garage',
          period: '2021',
          logo: { initials: 'G', src: '/logos/google.webp', alt: 'Google logo' },
        },
      ],
    },
    {
      id: 'awards',
      label: 'Awards',
      entries: [
        {
          title: 'Academic Excellence Scholarship',
          organization: 'Community of Madrid',
          period: '2024 and 2025',
          logo: { initials: 'CM', src: '/logos/ccaa_mad.webp', alt: 'Community of Madrid logo' },
        },
        {
          title: '1st Place - UAM Investors League',
          organization: 'Autonomous University of Madrid + IronIA Fintech',
          period: '2025',
          logo: { initials: 'LI', src: '/logos/fuam.png', alt: 'Autonomous University of Madrid Foundation logo' },
        },
        {
          title: '2nd Place - Cloud Agent Systems Hackathon',
          organization: 'Google Cloud + Diverger',
          period: '2025',
          logo: { initials: 'GC', src: '/logos/cloud.png', alt: 'Google Cloud logo' },
        },
        {
          title: 'Academic Recognition for High Achievement',
          organization: 'Colegio San José de Begoña',
          period: '2020 and 2021',
          logo: { initials: 'CSJ', src: '/logos/colegio.png', alt: 'Colegio San José de Begoña crest' },
        },
      ],
    },
  ],
};

export const profileProjectsEn: ProfileProject[] = [
  {
    title: 'Surrogate models for finding optimal inputs',
    excerpt: 'Gaussian processes, Bayesian optimization and Expected Improvement for finding strong configurations under a limited evaluation budget.',
    year: '2026',
    github: 'https://github.com/trentisiete/surrogate_models',
  },
  {
    title: 'Image generation with diffusion models',
    excerpt: 'Implementation of the SDE framework by Song et al.: denoising score matching training, samplers, conditional generation and imputation on CIFAR-10.',
    year: '2025',
    github: 'https://github.com/trentisiete/DiffusionImaGen',
    articleHref: '/en/articles/diffusion_models/',
  },
];
