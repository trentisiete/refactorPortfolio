export type AboutPlaylistPlatform = 'youtube' | 'spotify';
export type AboutMusicKind = 'playlist' | 'album';

export interface AboutPlaylist {
  platform: AboutPlaylistPlatform;
  kind: AboutMusicKind;
  id: string;
  title: string;
  description: string;
}

export const aboutPlaylists: AboutPlaylist[] = [
  {
    platform: 'youtube',
    kind: 'playlist',
    id: 'PLyi4gdcJFtgeRCwYNR_KtqdzUYSUUSYDk',
    title: 'A playlist to watch.',
    description: 'Videos and songs gathered in one slightly unruly queue.',
  },
  {
    platform: 'spotify',
    kind: 'playlist',
    id: '40NMQCa8DmUnHMKmvwq1fg',
    title: 'Puros Corridos Tumbados Viejo.',
    description: 'It is my duty to share these masterpieces with everyone.',
  },
  {
    platform: 'spotify',
    kind: 'album',
    id: '1weenld61qoidwYuZ1GESA',
    title: 'My first vinyl.',
    description: 'Lorena, my girlfriend, gifted me this jazz masterpiece on vinyl.',
  },
  {
    platform: 'spotify',
    kind: 'playlist',
    id: '2rTWmwS3pTkT4dcKDbQK3E',
    title: 'Laid back and enjoy life.',
    description: 'I stumbled upon this gem during a short commute with my fellow Mikel.',
  },
];
