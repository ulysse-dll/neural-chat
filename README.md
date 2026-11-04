# Neural Chat

Une interface web locale pour discuter avec tes modèles Ollama, avec en fond un « cerveau » 3D animé (Canvas 2D, sans librairie) qui réagit aux vrais tokens : un influx par token, intensité selon les tokens/s, couleur selon la confiance du modèle (logprobs).

Un seul fichier : `index.html`, zéro dépendance.

## Lancer

```bash
python3 -m http.server 8765
# puis ouvrir http://localhost:8765
```

En double-cliquant sur `index.html` (`file://`), Ollama refuse la connexion par défaut et la page passe en mode démo. Pour l'autoriser :

```bash
launchctl setenv OLLAMA_ORIGINS "null"   # puis relancer l'app Ollama
```

## Brancher la sphère ailleurs

```js
neural.pulse(3, 0.9)    // 3 influx, confiance 0..1 (rouge = sûr, jaune = incertain)
neural.think(true)      // mode réflexion
neural.setActivity(0.5) // activité 0..1
```

La config (URL Ollama, neurones, liaisons, prompt système) est en haut du script.
