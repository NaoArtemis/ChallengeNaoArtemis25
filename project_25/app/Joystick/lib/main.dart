import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(JoystickApp());
}

class JoystickApp extends StatelessWidget {
  const JoystickApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'NAO Robot Controller',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: JoystickPage(),
    );
  }
}

class JoystickPage extends StatefulWidget {
  const JoystickPage({super.key});

  @override
  _JoystickPageState createState() => _JoystickPageState();
}

class _JoystickPageState extends State<JoystickPage> {
  final String _serverAddress = '192.168.0.103:5010'; // IP + Port

  Future<void> _sendMovementCommand(String action) async {
    final Uri uri = Uri.parse('http://$_serverAddress/api/movement/$action');

    try {
      final response = await http.get(uri);

      if (response.statusCode == 200) {
        print('Comando $action inviato con successo');
      } else if (response.statusCode == 302) {
        print('Redirect ignorato per il comando: $action');
        // Non eseguire nessuna richiesta di follow-up
      } else {
        print('Errore nella risposta del server: ${response.statusCode}');
      }
    } catch (e) {
      print('Errore durante la connessione: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('NAO Robot Controller')),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ElevatedButton(
              onPressed: () => _sendMovementCommand('start'),
              child: const Text('↑ AVANTI'),
            ),
            const SizedBox(height: 10),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                ElevatedButton(
                  onPressed: () => _sendMovementCommand('left'),
                  child: const Text('← SINISTRA'),
                ),
                const SizedBox(width: 20),
                ElevatedButton(
                  onPressed: () => _sendMovementCommand('stop'),
                  style: ElevatedButton.styleFrom(backgroundColor: Colors.red),
                  child: Text('STOP'),
                ),
                const SizedBox(width: 20),
                ElevatedButton(
                  onPressed: () => _sendMovementCommand('right'),
                  child: const Text('DESTRA →'),
                ),
              ],
            ),
            const SizedBox(height: 10),
            ElevatedButton(
              onPressed: () => _sendMovementCommand('back'),
              child: const Text('↓ DIETRO'),
            ),
          ],
        ),
      ),
    );
  }
}