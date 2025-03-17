import 'package:flutter/material.dart';
import 'package:flutter_joystick/flutter_joystick.dart';
<<<<<<< Updated upstream
import 'package:http/http.dart' as http;
import 'dart:convert';
=======
>>>>>>> Stashed changes

void main() {
  runApp(MyApp());
}

<<<<<<< Updated upstream
class MyApp extends StatefulWidget {
  @override
  _MyAppState createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  String direction = "Fermo";

  void updateDirection(StickDragDetails details) {
    double x = details.x;
    double y = details.y;

    String newDirection;
    if (y < -0.5) {
      newDirection = "Avanti";
    } else if (y > 0.5) {
      newDirection = "Indietro";
    } else if (x > 0.5) {
      newDirection = "Destra";
    } else if (x < -0.5) {
      newDirection = "Sinistra";
    } else {
      newDirection = "Fermo";
    }

    if (newDirection != direction) {
      setState(() {
        direction = newDirection;
      });
      sendDirection(newDirection);
    }
  }

  void sendDirection(String direction) async {
    var url = Uri.parse('https://tuo-server.com/api/direction'); // Cambia con il tuo URL
    try {
      var response = await http.post(
        url,
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({"direction": direction}),
      );

      if (response.statusCode == 200) {
        print("✅ Direzione inviata con successo: $direction");
      } else {
        print("❌ Errore nell'invio: ${response.statusCode} - ${response.body}");
      }
    } catch (e) {
      print("⚠️ Errore di connessione: $e");
    }
=======
class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: JoystickScreen(),
    );
  } 
}

class JoystickScreen extends StatefulWidget {
  @override
  _JoystickScreenState createState() => _JoystickScreenState();
}

class _JoystickScreenState extends State<JoystickScreen> {
  String direction = "Fermo"; // Direzione iniziale

  void _updateDirection(StickDragDetails details) {
    double dx = details.x; // Offset orizzontale
    double dy = details.y; // Offset verticale

    setState(() {
      if (dy < -0.5) {
        direction = "Avanti";
      } else if (dy > 0.5) {
        direction = "Indietro";
      } else if (dx > 0.5) {
        direction = "Destra";
      } else if (dx < -0.5) {
        direction = "Sinistra";
      } else {
        direction = "Fermo";
      }
    });
>>>>>>> Stashed changes
  }

  @override
  Widget build(BuildContext context) {
<<<<<<< Updated upstream
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: Text('Joystick con Direzioni')),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text(
                'Direzione: $direction',
                style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
              ),
              SizedBox(height: 20),
              Joystick(
                listener: updateDirection,
              ),
            ],
          ),
=======
    return Scaffold(
      appBar: AppBar(title: Text('Joystick con Direzioni')),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(
              'Direzione: $direction',
              style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 20),
            Joystick(
              mode: JoystickMode.all,
              listener: _updateDirection, // Ora usa il tipo corretto
            ),
          ],
>>>>>>> Stashed changes
        ),
      ),
    );
  }
}
