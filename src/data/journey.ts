import type { Lang } from '../i18n/utils';

export type JourneyLaneId = 'experience' | 'education' | 'awards';
export type JourneyCardSize = 'compact' | 'medium' | 'tall';
export type JourneyTrack = 1 | 2 | 3;

export interface JourneyMilestoneData {
  title: string;
  offset: number;
  text: string;
}

export interface JourneyTimelineItem {
  id: string;
  lane: JourneyLaneId;
  track: JourneyTrack;
  start: number;
  end?: number;
  title: string;
  place: string;
  description: string;
  size: JourneyCardSize;
  sticky?: boolean;
  hidePlace?: boolean;
  throughCurrent?: boolean;
  stack?: string;
  milestones?: JourneyMilestoneData[];
}

interface JourneyItemLayout {
  id: string;
  lane: JourneyLaneId;
  track: JourneyTrack;
  start: number;
  end?: number;
  size: JourneyCardSize;
  sticky?: boolean;
  hidePlace?: boolean;
  throughCurrent?: boolean;
  stack?: string;
  milestoneOffsets?: number[];
}

interface JourneyItemCopy {
  title: string;
  place: string;
}

interface JourneyLocaleCopy {
  lanes: Record<JourneyLaneId, string>;
  timelineLabel: string;
  lanesLabel: string;
  yearLabel: string;
  today: string;
  items: Record<string, JourneyItemCopy>;
}

export interface JourneyTimelineData {
  years: number[];
  lanes: Record<JourneyLaneId, string>;
  timelineLabel: string;
  lanesLabel: string;
  yearLabel: string;
  today: string;
  items: JourneyTimelineItem[];
}

const years = [2020, 2021, 2022, 2023, 2024, 2025, 2026];

const itemLayouts: JourneyItemLayout[] = [
  {
    id: 'academic-recognition-2020',
    lane: 'awards',
    track: 1,
    start: 2020,
    size: 'compact',
  },
  {
    id: 'academic-recognition-2021',
    lane: 'awards',
    track: 1,
    start: 2021,
    size: 'compact',
  },
  {
    id: 'digital-marketing',
    lane: 'education',
    track: 1,
    start: 2021,
    size: 'compact',
  },
  {
    id: 'personal-training',
    lane: 'education',
    track: 2,
    start: 2021,
    end: 2023,
    size: 'medium',
    sticky: true,
  },
  {
    id: 'dripinlocker',
    lane: 'experience',
    track: 1,
    start: 2021,
    end: 2024,
    size: 'medium',
    sticky: true,
  },
  {
    id: 'saona',
    lane: 'experience',
    track: 2,
    start: 2022,
    size: 'compact',
  },
  {
    id: 'data-science-degree',
    lane: 'education',
    track: 1,
    start: 2022,
    end: 2026,
    size: 'tall',
    sticky: true,
    hidePlace: true,
    throughCurrent: true,
    milestoneOffsets: [1, 2, 3, 4],
  },
  {
    id: 'seif',
    lane: 'experience',
    track: 2,
    start: 2023,
    end: 2024,
    size: 'medium',
  },
  {
    id: 'uam-emprende',
    lane: 'education',
    track: 2,
    start: 2023,
    size: 'compact',
  },
  {
    id: 'webmaster',
    lane: 'experience',
    track: 2,
    start: 2024,
    end: 2025,
    size: 'medium',
  },
  {
    id: 'data-analytics-google',
    lane: 'education',
    track: 2,
    start: 2024,
    size: 'compact',
  },
  {
    id: 'data-engineering-nanjing',
    lane: 'education',
    track: 3,
    start: 2024,
    size: 'compact',
  },
  {
    id: 'academic-scholarship-2024',
    lane: 'awards',
    track: 1,
    start: 2024,
    size: 'compact',
  },
  {
    id: 'gdg',
    lane: 'experience',
    track: 1,
    start: 2025,
    end: 2026,
    size: 'medium',
    sticky: true,
    throughCurrent: true,
  },
  {
    id: 'cloud-engineering',
    lane: 'education',
    track: 2,
    start: 2025,
    end: 2026,
    size: 'medium',
    sticky: true,
    throughCurrent: true,
  },
  {
    id: 'data-engineering-google',
    lane: 'education',
    track: 3,
    start: 2025,
    size: 'compact',
    stack: 'education-2025',
  },
  {
    id: 'investors-course',
    lane: 'education',
    track: 3,
    start: 2025,
    size: 'compact',
    stack: 'education-2025',
  },
  {
    id: 'investors-award',
    lane: 'awards',
    track: 1,
    start: 2025,
    size: 'compact',
    stack: 'awards-2025',
  },
  {
    id: 'academic-scholarship-2025',
    lane: 'awards',
    track: 1,
    start: 2025,
    size: 'compact',
    stack: 'awards-2025',
  },
  {
    id: 'cloud-hackathon',
    lane: 'awards',
    track: 1,
    start: 2025,
    size: 'compact',
    stack: 'awards-2025',
  },
  {
    id: 'naudit',
    lane: 'experience',
    track: 2,
    start: 2026,
    size: 'compact',
  },
];

const localeCopy: Record<Lang, JourneyLocaleCopy> = {
  es: {
    lanes: {
      experience: 'Experiencia',
      education: 'Educación',
      awards: 'Premios',
    },
    timelineLabel: 'Cronología de la trayectoria',
    lanesLabel: 'Secciones de la cronología',
    yearLabel: 'Año',
    today: 'Hoy',
    items: {
      'academic-recognition-2020': {
        title: 'Reconocimiento académico por alto rendimiento',
        place: 'Colegio San José de Begoña',
      },
      'academic-recognition-2021': {
        title: 'Reconocimiento académico por alto rendimiento',
        place: 'Colegio San José de Begoña',
      },
      'digital-marketing': {
        title: 'Fundamentos de Marketing Digital',
        place: 'Google Digital Garage',
      },
      'personal-training': {
        title: 'Formación en Entrenamiento Personal',
        place: 'Trainologym',
      },
      dripinlocker: {
        title: 'Emprendedor independiente',
        place: 'DripInLocker',
      },
      saona: {
        title: 'Camarero',
        place: 'Saona',
      },
      'data-science-degree': {
        title: 'Grado en Ciencia e Ingeniería de Datos',
        place: 'Universidad Autónoma de Madrid',
      },
      seif: {
        title: 'Miembro de la junta directiva',
        place: 'Club de Seguridad Informática (SEIF UAM)',
      },
      'uam-emprende': {
        title: 'Programa UAM Emprende',
        place: 'Universidad Autónoma de Madrid',
      },
      webmaster: {
        title: 'Webmaster',
        place: 'AUTOCENTER - VULCASAN SL',
      },
      'data-analytics-google': {
        title: 'Data Analytics',
        place: 'Google',
      },
      'data-engineering-nanjing': {
        title: 'Data Engineering',
        place: 'Nanjing University',
      },
      'academic-scholarship-2024': {
        title: 'Beca de Excelencia Académica',
        place: 'Comunidad de Madrid',
      },
      gdg: {
        title: 'Presidente y fundador',
        place: 'Google Developer Group on Campus UAM',
      },
      'cloud-engineering': {
        title: 'Formación en Cloud Engineering',
        place: 'Google Cloud',
      },
      'data-engineering-google': {
        title: 'Data Engineering',
        place: 'Google',
      },
      'investors-course': {
        title: 'Liga de Inversores: Reta tu Estrategia de Carteras en Vivo',
        place: 'UAM + IronIA Fintech',
      },
      'investors-award': {
        title: '1.er puesto - Liga de Inversores UAM',
        place: 'Universidad Autónoma de Madrid + IronIA Fintech',
      },
      'academic-scholarship-2025': {
        title: 'Beca de Excelencia Académica',
        place: 'Comunidad de Madrid',
      },
      'cloud-hackathon': {
        title: '2.º puesto - Hackathon de Sistemas de Agentes Cloud',
        place: 'Google Cloud + Diverger',
      },
      naudit: {
        title: 'Data Scientist Intern',
        place: 'Naudit HPCN',
      },
    },
  },
  en: {
    lanes: {
      experience: 'Experience',
      education: 'Education',
      awards: 'Awards',
    },
    timelineLabel: 'Journey timeline',
    lanesLabel: 'Timeline sections',
    yearLabel: 'Year',
    today: 'Today',
    items: {
      'academic-recognition-2020': {
        title: 'Academic Recognition for High Achievement',
        place: 'Colegio San José de Begoña',
      },
      'academic-recognition-2021': {
        title: 'Academic Recognition for High Achievement',
        place: 'Colegio San José de Begoña',
      },
      'digital-marketing': {
        title: 'Fundamentals of Digital Marketing',
        place: 'Google Digital Garage',
      },
      'personal-training': {
        title: 'Personal Training Education',
        place: 'Trainologym',
      },
      dripinlocker: {
        title: 'Independent Entrepreneur',
        place: 'DripInLocker',
      },
      saona: {
        title: 'Waiter',
        place: 'Saona',
      },
      'data-science-degree': {
        title: "Bachelor's Degree in Data Science and Engineering",
        place: 'Autonomous University of Madrid',
      },
      seif: {
        title: 'Board Member',
        place: 'Computer Security Club (SEIF UAM)',
      },
      'uam-emprende': {
        title: 'UAM Emprende Program',
        place: 'Autonomous University of Madrid',
      },
      webmaster: {
        title: 'Webmaster',
        place: 'AUTOCENTER - VULCASAN SL',
      },
      'data-analytics-google': {
        title: 'Data Analytics',
        place: 'Google',
      },
      'data-engineering-nanjing': {
        title: 'Data Engineering',
        place: 'Nanjing University',
      },
      'academic-scholarship-2024': {
        title: 'Academic Excellence Scholarship',
        place: 'Community of Madrid',
      },
      gdg: {
        title: 'President & Founder',
        place: 'Google Developer Group on Campus UAM',
      },
      'cloud-engineering': {
        title: 'Cloud Engineering Training',
        place: 'Google Cloud',
      },
      'data-engineering-google': {
        title: 'Data Engineering',
        place: 'Google',
      },
      'investors-course': {
        title: 'Investors League: Challenge Your Portfolio Strategy Live',
        place: 'UAM + IronIA Fintech',
      },
      'investors-award': {
        title: '1st Place - UAM Investors League',
        place: 'Autonomous University of Madrid + IronIA Fintech',
      },
      'academic-scholarship-2025': {
        title: 'Academic Excellence Scholarship',
        place: 'Community of Madrid',
      },
      'cloud-hackathon': {
        title: '2nd Place - Cloud Agent Systems Hackathon',
        place: 'Google Cloud + Diverger',
      },
      naudit: {
        title: 'Data Scientist Intern',
        place: 'Naudit HPCN',
      },
    },
  },
  de: {
    lanes: {
      experience: 'Berufserfahrung',
      education: 'Ausbildung',
      awards: 'Auszeichnungen',
    },
    timelineLabel: 'Zeitleiste des Werdegangs',
    lanesLabel: 'Bereiche der Zeitleiste',
    yearLabel: 'Jahr',
    today: 'Heute',
    items: {
      'academic-recognition-2020': {
        title: 'Akademische Anerkennung für herausragende Leistungen',
        place: 'Colegio San José de Begoña',
      },
      'academic-recognition-2021': {
        title: 'Akademische Anerkennung für herausragende Leistungen',
        place: 'Colegio San José de Begoña',
      },
      'digital-marketing': {
        title: 'Grundlagen des digitalen Marketings',
        place: 'Google Digital Garage',
      },
      'personal-training': {
        title: 'Ausbildung zum Personal Trainer',
        place: 'Trainologym',
      },
      dripinlocker: {
        title: 'Selbstständiger Unternehmer',
        place: 'DripInLocker',
      },
      saona: {
        title: 'Kellner',
        place: 'Saona',
      },
      'data-science-degree': {
        title: 'Bachelorstudium in Data Science and Engineering',
        place: 'Autonome Universität Madrid',
      },
      seif: {
        title: 'Vorstandsmitglied',
        place: 'Club für IT-Sicherheit (SEIF UAM)',
      },
      'uam-emprende': {
        title: 'Programm UAM Emprende',
        place: 'Autonome Universität Madrid',
      },
      webmaster: {
        title: 'Webmaster',
        place: 'AUTOCENTER - VULCASAN SL',
      },
      'data-analytics-google': {
        title: 'Data Analytics',
        place: 'Google',
      },
      'data-engineering-nanjing': {
        title: 'Data Engineering',
        place: 'Nanjing University',
      },
      'academic-scholarship-2024': {
        title: 'Stipendium für akademische Exzellenz',
        place: 'Comunidad de Madrid',
      },
      gdg: {
        title: 'Präsident und Gründer',
        place: 'Google Developer Group on Campus UAM',
      },
      'cloud-engineering': {
        title: 'Weiterbildung in Cloud Engineering',
        place: 'Google Cloud',
      },
      'data-engineering-google': {
        title: 'Data Engineering',
        place: 'Google',
      },
      'investors-course': {
        title: 'Investorenliga: Stelle deine Portfoliostrategie live auf die Probe',
        place: 'UAM + IronIA Fintech',
      },
      'investors-award': {
        title: '1. Platz - Investorenliga der UAM',
        place: 'Autonome Universität Madrid + IronIA Fintech',
      },
      'academic-scholarship-2025': {
        title: 'Stipendium für akademische Exzellenz',
        place: 'Comunidad de Madrid',
      },
      'cloud-hackathon': {
        title: '2. Platz - Hackathon für Cloud-Agentensysteme',
        place: 'Google Cloud + Diverger',
      },
      naudit: {
        title: 'Data Scientist Intern',
        place: 'Naudit HPCN',
      },
    },
  },
};

export const getJourneyTimelineData = (lang: Lang): JourneyTimelineData => {
  const copy = localeCopy[lang];

  return {
    years,
    lanes: copy.lanes,
    timelineLabel: copy.timelineLabel,
    lanesLabel: copy.lanesLabel,
    yearLabel: copy.yearLabel,
    today: copy.today,
    items: itemLayouts.map(item => ({
      ...item,
      ...copy.items[item.id],
      description: 'INFO',
      milestones: item.milestoneOffsets?.map(offset => ({
        offset,
        title: 'INFO',
        text: 'INFO',
      })),
    })),
  };
};
