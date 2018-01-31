# Anteckningar

## LSA

Problemet ligger i att vi vill fånga semantik separat för varje dokument, men göra dimensionalitetsreduktionen på 
features gemensamma för alla dokument.

Om det överhuvudtaget funkar tänker jag såhär:

- Skapa dictionary gemensam för alla dokument. Detta ger oss samma features för alla dokument.
- Skapa ett corpus för varje dokument.
- Utför LSA på varje enskilt corpus men med samma antal topics för alla corpora.