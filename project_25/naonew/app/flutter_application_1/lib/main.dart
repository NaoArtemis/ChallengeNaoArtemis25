import 'package:flutter/material.dart';
import 'package:flutter_joystick/flutter_joystick.dart';

void main() {
  runApp(MyApp());
}

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
  }

  @override
  Widget build(BuildContext context) {
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
        ),
      ),
    );
  }
}
