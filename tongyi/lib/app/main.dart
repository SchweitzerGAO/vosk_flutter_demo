import 'dart:io';

import 'package:flutter/material.dart';
import 'package:vosk_flutter/vosk_flutter.dart';
import 'package:auto_size_text/auto_size_text.dart';
import 'package:flutter_vibrate/flutter_vibrate.dart';


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
  String _asrModelName = 'vosk-model-small-cn-0.22.zip';
  String _srcVal = '中文';
  String _tgtVal = '中文';

  final _sampleRate = 16000;

  final _vosk = VoskFlutterPlugin.instance();
  final _modelLoader = ModelLoader();

  String? _error;
  Model? _model;
  Recognizer? _recognizer;
  SpeechService? _speechService;

  bool _recognitionStarted = false;

  final _availableLangs = [
    '中文',
    '英语',
    '日语',
    '法语',
    '俄语',
    '西班牙语',
    '葡萄牙语'
  ];


  final _modelList = [
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

  void _swapLang() {

  }


  @override
  void initState() {
    super.initState();
    _loadModel(_asrModelName);
  }

  Widget _buildMainPage(){
   return Scaffold(
    body: SafeArea(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: <Widget>[
          SizedBox(
            width: 300,
            height: 500,
            child: StreamBuilder(
                stream: _speechService!.onPartial(),
                builder: (context, snapshot) => AutoSizeText(
                    snapshot.data.toString(),
                    style: const TextStyle(fontSize: 25),
                    maxLines: 3,
                    )),
          ),
          FloatingActionButton.large(
              onPressed:() async {
                Vibrate.feedback(FeedbackType.selection);
                if (_recognitionStarted) {
                  await _speechService!.stop();
                }
                else {
                  await _speechService!.start();
                }
                setState(() => _recognitionStarted = !_recognitionStarted);
              },
              backgroundColor: _recognitionStarted ? Colors.blue : Colors.black87,
              child: const Icon(Icons.mic)
          ),
          _buildSelectBox(),
        ],

      ),
    ),
   );
  }

  Widget _buildSelectBox() {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 8.0),
      child: ButtonBar(
        children: <Widget>[
          DropdownButton(
              value: _srcVal,
              items: _availableLangs.map<DropdownMenuItem<String>> ((String value) {
                return DropdownMenuItem<String>(
                  value: value,
                  child: Text(value),
                );
              }).toList(),
              onChanged: (String? value) {
                setState(() {
                    _srcVal = value!;
                });
                _loadModel(_modelList[_availableLangs.indexOf(_srcVal)]);
              }
          ),
          IconButton(
              onPressed: _swapLang,
              icon: const Icon(Icons.swap_horiz_outlined)
          ),
          DropdownButton(
              value: _tgtVal,
              items: _availableLangs.map<DropdownMenuItem<String>> ((String value) {
                return DropdownMenuItem<String>(
                  value: value,
                  child: Text(value),
                );
              }).toList(),
              onChanged: (String? value) {
                setState(() {
                  _tgtVal = value!;
                });
              }
          ),

        ],


      ),
    );
  }



  @override
  Widget build(BuildContext context) {
    if (_error != null) {
      return Scaffold(
          body: Center(child: Text("Error: $_error")));
    } else if (_model == null) {
      return const Scaffold(
          body: Center(child: Text("Loading model...")));
    } else if (Platform.isAndroid && _speechService == null) {
      return const Scaffold(
        body: Center(
          child: Text("Initializing speech service..."),
        ),
      );
    } else {
      return _buildMainPage();
    }
  }

}