import 'dart:io';

import 'package:flutter/material.dart';
import 'package:record/record.dart';
import 'package:vosk_flutter/vosk_flutter.dart';

enum Language {
  cn,
  en,
  ja,
  fr,
  ru,
  es,
  pt,
  ar
}

void main() {
  runApp(const MainApp());
}


class MainApp extends StatelessWidget {
  const MainApp({super.key});
  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      home: MainPage(),
    );
  }

}

class MainPage extends StatefulWidget {
  const MainPage({super.key});

  @override
  State<StatefulWidget> createState() => _MainPageState();
}

class _MainPageState extends State<MainPage> {
  String _modelName = '';
  final _sampleRate = 16000;

  final _vosk = VoskFlutterPlugin.instance();
  final _modelLoader = ModelLoader();
  final _recorder = Record();

  String? _recognitionResult;
  String? _error;
  Model? _model;
  Recognizer? _recognizer;
  SpeechService? _speechService;

  bool _recognitionStarted = false;
  final modelList = [
    'vosk-model-small-cn-0.22.zip',
    'vosk-model-small-en-us-0.15.zip',
    'vosk-model-small-ja-0.22.zip',
    'vosk-model-small-fr-0.22.zip',
    'vosk-model-small-ru-0.22.zip',
    'vosk-model-small-es-0.42.zip',
    'vosk-model-small-pt-0.22.zip',

  ];

  void _loadModel(modelName){
    _modelLoader.loadFromAssets(modelName)
        .then((modelPath) => _vosk.createModel(modelPath))
        .then((model) => setState(() => _model = model))
        .then((_) => _vosk.createRecognizer(model: _model!, sampleRate: _sampleRate))
        .then((value) => _recognizer = value)
        .then((recognizer) {
      if(Platform.isAndroid) {
        _vosk.initSpeechService(_recognizer!)
            .then((speechService) =>
            setState(() => _speechService = speechService))
            .catchError((e) => setState(() => _error = e.toString()));
      }
    }).catchError((e) {
      setState(() {
        _error = e.toString();
      });
      return null;
    });
  }

  @override
  void initState() {
    super.initState();

  }



  @override
  Widget build(BuildContext context) {
    // TODO: implement build
    throw UnimplementedError();
  }

}