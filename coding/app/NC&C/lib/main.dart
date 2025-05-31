import 'package:flutter/material.dart';
import 'package:google_sign_in/google_sign_in.dart';
import 'package:http/http.dart' as http;
import 'package:pedometer/pedometer.dart';
import 'package:geolocator/geolocator.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'dart:convert';
import 'dart:async';

void main() {
  runApp(const HealthApp());
}

String? globalUserId;

class HealthApp extends StatelessWidget {
  const HealthApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(primarySwatch: Colors.blue, useMaterial3: true),
      home: const LoginPage(),
    );
  }
}

class User {
  final String id;
  final String firstName;
  final String lastName;
  final String username;
  final String password;
  int? heartRate;
  int? stepCount;
  double? speed;
  Position? position;

  User({
    required this.id,
    required this.firstName,
    required this.lastName,
    required this.username,
    required this.password,
    this.heartRate,
    this.stepCount,
    this.speed,
    this.position,
  });

  Map<String, dynamic> toJson() => {
        'id': id,
        'firstName': firstName,
        'lastName': lastName,
        'username': username,
        'password': password,
        'heartRate': heartRate,
        'stepCount': stepCount,
        'speed': speed,
        'latitude': position?.latitude,
        'longitude': position?.longitude,
      };

  factory User.fromJson(Map<String, dynamic> json) => User(
        id: json['id'],
        firstName: json['firstName'],
        lastName: json['lastName'],
        username: json['username'],
        password: json['password'],
        heartRate: json['heartRate'],
        stepCount: json['stepCount'],
        speed: json['speed'],
        position: (json['latitude'] != null && json['longitude'] != null)
            ? Position(
                latitude: json['latitude'],
                longitude: json['longitude'],
                timestamp: DateTime.now(),
                accuracy: 0,
                altitude: 0,
                heading: 0,
                speed: json['speed'] ?? 0,
                speedAccuracy: 0,
                altitudeAccuracy: 0,
                headingAccuracy: 0,
              )
            : null,
      );
}

class ServerConfig {
  static String serverIP = "192.168.0.103";
  static const int serverPort = 5010;

  static Future<void> updateServerIP(String newIP) async {
    serverIP = newIP;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('serverIP', newIP);
  }

  static Future<void> loadServerIP() async {
    final prefs = await SharedPreferences.getInstance();
    serverIP = prefs.getString('serverIP') ?? "192.168.0.103";
  }
}

class AuthService {
  static List<User> users = [];

  static Future<void> loadUsers() async {
    final prefs = await SharedPreferences.getInstance();
    final List<String>? userJsonList = prefs.getStringList('users');
    if (userJsonList != null) {
      users = userJsonList
          .map((userJson) => User.fromJson(json.decode(userJson)))
          .toList();
    }
  }

  static Future<void> saveUsers() async {
    final prefs = await SharedPreferences.getInstance();
    final List<String> userJsonList =
        users.map((user) => json.encode(user.toJson())).toList();
    await prefs.setStringList('users', userJsonList);
  }

  static Future<void> register(User newUser) async {
    await ServerConfig.loadServerIP();
    final url = Uri.parse(
        'http://${ServerConfig.serverIP}:${ServerConfig.serverPort}/register');

    final body = json.encode(newUser.toJson());

    try {
      final response = await http.post(
        url,
        headers: {
          'Content-Type': 'application/json',
        },
        body: body,
      );

      if (response.statusCode == 200) {
        users.add(newUser);
        await saveUsers();
        print('Utente registrato con successo!');
      } else {
        print('Errore durante la registrazione: ${response.statusCode}');
      }
    } catch (error) {
      print('Errore di connessione: $error');
    }
  }

  static bool login(String username, String password) {
    return users.any((user) =>
        user.username == username && user.password == password);
  }

  static Future<void> sendUserDataToServer(User user) async {
    await ServerConfig.loadServerIP();
    final url = Uri.parse(
        'http://${ServerConfig.serverIP}:${ServerConfig.serverPort}/api/app/dati/${user.id}');

    final body = jsonEncode({
      'id_player': user.id,
      'bpm': user.heartRate ?? 0,
      'passi': user.stepCount ?? 0,
      'velocita': user.speed ?? 0.0,
    });

    try {
      final response = await http.post(
        url,
        headers: {
          'Content-Type': 'application/json',
        },
        body: body,
      );

      if (response.statusCode == 200) {
        print('✅ Dati inviati con successo: ${response.body}');
      } else {
        print('❌ Errore [${response.statusCode}]: ${response.body}');
      }
    } catch (error) {
      print('⚠️ Eccezione durante la richiesta: $error');
    }
  }
}

class LoginPage extends StatefulWidget {
  const LoginPage({super.key});

  @override
  State<LoginPage> createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  final _usernameController = TextEditingController();
  final _passwordController = TextEditingController();

  void _login() async {
    if (_usernameController.text.isEmpty || _passwordController.text.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Inserisci username e password')),
      );
      return;
    }

    await ServerConfig.loadServerIP();

    final url = Uri.parse(
        'http://${ServerConfig.serverIP}:${ServerConfig.serverPort}/api/app/utente/${_usernameController.text}');

    try {
      final response = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: json.encode({
          'username': _usernameController.text,
          'password': _passwordController.text,
        }),
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);

        if (data['code'] == 200) {
          final userData = data['data'];

          globalUserId = userData['id'].toString();  // <-- Salvo qui l'id

          final user = User(
            id: globalUserId!,
            firstName: userData['nome'] ?? '',
            lastName: userData['cognome'] ?? '',
            username: _usernameController.text,
            password: _passwordController.text,
          );

          Navigator.pushReplacement(
            context,
            MaterialPageRoute(builder: (context) => HealthScreen(user: user)),
          );
        } else {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text(data['message'] ?? 'Errore sconosciuto')),
          );
        }
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Errore server: ${response.statusCode}')),
        );
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Errore di rete: $e')),
      );
    }
  }

  void _navigateToRegister() {
    Navigator.push(
      context,
      MaterialPageRoute(builder: (context) => const RegisterPage()),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color.fromARGB(255, 243, 245, 246),
      body: Center(
        child: SingleChildScrollView(
          child: Container(
            padding: const EdgeInsets.all(16.0),
            margin: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.9),
              borderRadius: BorderRadius.circular(20),
              boxShadow: const [
                BoxShadow(
                  color: Colors.black26,
                  offset: Offset(0, 4),
                  blurRadius: 8,
                ),
              ],
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Image.asset(
                  'assets2/icons/logo_v3.png',
                  width: 300,
                  fit: BoxFit.contain,
                ),
                const Text(
                  'Benvenuto!',
                  style: TextStyle(
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                    color: Colors.blue,
                  ),
                ),
                const SizedBox(height: 20),
                TextField(
                  controller: _usernameController,
                  decoration: const InputDecoration(
                    labelText: 'Nome utente',
                    labelStyle: TextStyle(fontWeight: FontWeight.bold),
                    border: OutlineInputBorder(),
                    prefixIcon: Icon(Icons.person),
                  ),
                ),
                const SizedBox(height: 20),
                TextField(
                  controller: _passwordController,
                  obscureText: true,
                  decoration: const InputDecoration(
                    labelText: 'Password',
                    labelStyle: TextStyle(fontWeight: FontWeight.bold),
                    border: OutlineInputBorder(),
                    prefixIcon: Icon(Icons.lock),
                  ),
                ),
                const SizedBox(height: 30),
                ElevatedButton(
                  onPressed: _login,
                  style: ElevatedButton.styleFrom(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 50, vertical: 15),
                    backgroundColor: Colors.blue,
                  ),
                  child: const Text('Accedi'),
                ),
                const SizedBox(height: 20),
                TextButton(
                  onPressed: _navigateToRegister,
                  child: const Text(
                    'Non hai un account? Registrati',
                    style: TextStyle(color: Colors.blue),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class RegisterPage extends StatefulWidget {
  const RegisterPage({super.key});

  @override
  State<RegisterPage> createState() => _RegisterPageState();
}

class _RegisterPageState extends State<RegisterPage> {
  final _idController = TextEditingController();
  final _firstNameController = TextEditingController();
  final _lastNameController = TextEditingController();
  final _usernameController = TextEditingController();
  final _passwordController = TextEditingController();

  void _register() async {
    if (_idController.text.isEmpty ||
        _firstNameController.text.isEmpty ||
        _lastNameController.text.isEmpty ||
        _usernameController.text.isEmpty ||
        _passwordController.text.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Tutti i campi sono obbligatori!')),
      );
      return;
    }

    bool userExists = AuthService.users.any((user) =>
        user.id == _idController.text || user.username == _usernameController.text);

    if (userExists) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('ID o Nome utente già esistente!')),
      );
      return;
    }

    final newUser = User(
      id: _idController.text,
      firstName: _firstNameController.text,
      lastName: _lastNameController.text,
      username: _usernameController.text,
      password: _passwordController.text,
    );

    await AuthService.register(newUser);

    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Registrazione completata!')),
    );

    Navigator.pop(context);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color.fromARGB(255, 243, 245, 246),
      body: Center(
        child: SingleChildScrollView(
          child: Container(
            padding: const EdgeInsets.all(16.0),
            margin: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.9),
              borderRadius: BorderRadius.circular(20),
              boxShadow: const [
                BoxShadow(
                  color: Colors.black26,
                  offset: Offset(0, 4),
                  blurRadius: 8,
                ),
              ],
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Text(
                  'Registrati',
                  style: TextStyle(
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                    color: Colors.blue,
                  ),
                ),
                const SizedBox(height: 20),
                TextField(
                  controller: _idController,
                  decoration: const InputDecoration(
                    labelText: 'ID',
                    border: OutlineInputBorder(),
                    prefixIcon: Icon(Icons.badge),
                  ),
                ),
                const SizedBox(height: 16),
                TextField(
                  controller: _firstNameController,
                  decoration: const InputDecoration(
                    labelText: 'Nome',
                    border: OutlineInputBorder(),
                    prefixIcon: Icon(Icons.person_outline),
                  ),
                ),
                const SizedBox(height: 16),
                TextField(
                  controller: _lastNameController,
                  decoration: const InputDecoration(
                    labelText: 'Cognome',
                    border: OutlineInputBorder(),
                    prefixIcon: Icon(Icons.person_outline),
                  ),
                ),
                const SizedBox(height: 16),
                TextField(
                  controller: _usernameController,
                  decoration: const InputDecoration(
                    labelText: 'Nome utente',
                    border: OutlineInputBorder(),
                    prefixIcon: Icon(Icons.person),
                  ),
                ),
                const SizedBox(height: 16),
                TextField(
                  controller: _passwordController,
                  obscureText: true,
                  decoration: const InputDecoration(
                    labelText: 'Password',
                    border: OutlineInputBorder(),
                    prefixIcon: Icon(Icons.lock),
                  ),
                ),
                const SizedBox(height: 30),
                ElevatedButton(
                  onPressed: _register,
                  style: ElevatedButton.styleFrom(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 50, vertical: 15),
                    backgroundColor: Colors.blue,
                  ),
                  child: const Text('Registrati'),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class ProfilePage extends StatelessWidget {
  final User user;
  const ProfilePage({super.key, required this.user});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Profilo Utente'),
      ),
      backgroundColor: const Color.fromARGB(255, 243, 245, 246),
      body: Center(
        child: SingleChildScrollView(
          child: Container(
            padding: const EdgeInsets.all(16.0),
            margin: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.9),
              borderRadius: BorderRadius.circular(20),
              boxShadow: const [
                BoxShadow(
                  color: Colors.black26,
                  offset: Offset(0, 4),
                  blurRadius: 8,
                ),
              ],
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('ID: ${user.id}', style: const TextStyle(fontSize: 18)),
                const SizedBox(height: 10),
                Text('Nome: ${user.firstName}', style: const TextStyle(fontSize: 18)),
                const SizedBox(height: 10),
                Text('Cognome: ${user.lastName}', style: const TextStyle(fontSize: 18)),
                const SizedBox(height: 10),
                Text('Nome Utente: ${user.username}', style: const TextStyle(fontSize: 18)),
                const SizedBox(height: 10),
                Text('Password: ${user.password}', style: const TextStyle(fontSize: 18)),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class HealthScreen extends StatefulWidget {
  final User user;
  const HealthScreen({super.key, required this.user});

  @override
  State<HealthScreen> createState() => _HealthScreenState();
}

class _HealthScreenState extends State<HealthScreen> {
  int _stepCount = 0;
  double? _speed;
  double? _latitude;
  double? _longitude;
  int? _heartRate;
  String _heartLog = "Recuperando dati da Google Fit...";
  late Timer _timer;

  @override
  void initState() {
    super.initState();
    _initServices();

    _timer = Timer.periodic(const Duration(minutes: 1), (timer) {
      _getHeartRateFromGoogleFit();
    });

    _getHeartRateFromGoogleFit();
  }

  @override
  void dispose() {
    _timer.cancel();
    super.dispose();
  }

  void _getStepCount() {
    Pedometer.stepCountStream.listen((stepCount) {
      setState(() {
        _stepCount = stepCount.steps;
      });
    });
  }

  Future<void> _getLocation() async {
    bool serviceEnabled = await Geolocator.isLocationServiceEnabled();
    if (!serviceEnabled) return;

    LocationPermission permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
      if (permission != LocationPermission.whileInUse) return;
    }

    Geolocator.getPositionStream().listen((Position position) {
      setState(() {
        _speed = position.speed;
        _latitude = position.latitude;
        _longitude = position.longitude;
      });
    });
  }

  Future<void> _sendDataToServer({
    required String idPlayer,
    required int bpm,
    required int passi,
    required double velocita,
  }) async {
    await ServerConfig.loadServerIP();
    final url = Uri.parse(
        'http://${ServerConfig.serverIP}:${ServerConfig.serverPort}/api/app/dati/$idPlayer');

    try { 
      final response = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'id_player': idPlayer,
          'bpm': bpm,
          'passi': passi,
          'velocita': velocita,
        }),
      );

      if (response.statusCode == 200) {
        print('✅ Dati inviati con successo: ${response.body}');
      } else {
        print('❌ Errore [${response.statusCode}]: ${response.body}');
      }
    } catch (e) {
      print('⚠️ Eccezione durante la richiesta: $e');
    }
  }

  Future<void> _getHeartRateFromGoogleFit() async {
    setState(() {
      _heartLog = "🔄 Avvio autenticazione con Google Fit...";
    });

    try {
      final GoogleSignIn googleSignIn = GoogleSignIn(
        scopes: [
          'https://www.googleapis.com/auth/fitness.heart_rate.read',
          'email',
        ],
      );

      final GoogleSignInAccount? account = await googleSignIn.signIn();
      if (account == null) {
        setState(() {
          _heartLog = "❌ Accesso annullato dall'utente.";
        });
        return;
      }

      final authHeaders = await account.authHeaders;
      final client = http.Client();

      final now = DateTime.now();
      final startTime = now.subtract(const Duration(hours: 1));
      final startMillis = startTime.millisecondsSinceEpoch;
      final endMillis = now.millisecondsSinceEpoch;

      final uri = Uri.parse(
          'https://www.googleapis.com/fitness/v1/users/me/dataset:aggregate');

      final response = await client.post(
        uri,
        headers: {
          'Authorization': authHeaders['Authorization']!,
          'Content-Type': 'application/json',
        },
        body: json.encode({
          "aggregateBy": [
            {
              "dataTypeName": "com.google.heart_rate.bpm",
              "dataSourceId":
                  "derived:com.google.heart_rate.bpm:com.google.android.gms:merge_heart_rate_bpm"
            }
          ],
          "bucketByTime": {"durationMillis": 60000},
          "startTimeMillis": startMillis,
          "endTimeMillis": endMillis,
        }),
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        final buckets = data['bucket'] as List<dynamic>;

        int? latestHeartRate;

        for (var bucket in buckets.reversed) {
          final dataset = bucket['dataset'][0];
          final points = dataset['point'] as List<dynamic>;

          if (points.isNotEmpty) {
            final value = points.last['value'][0]['fpVal'];
            latestHeartRate = value.toInt();
            break;
          }
        }

        setState(() {
          _heartRate = latestHeartRate;
          _heartLog = latestHeartRate != null
              ? "💓 Battito: $latestHeartRate bpm"
              : "❌ Nessun dato di battito recente trovato.";
        });

        if (latestHeartRate != null) {
          await _sendDataToServer(
            idPlayer: widget.user.id,
            bpm: latestHeartRate,
            passi: _stepCount,
            velocita: _speed ?? 0.0,
          );
        }
      } else {
        setState(() {
          _heartLog = "❌ Errore Google Fit: ${response.statusCode}";
        });
      }
    } catch (e) {
      setState(() {
        _heartLog = "❌ Errore: $e";
      });
    }
  }

  void _initServices() {
    _getStepCount();
    _getLocation();
    _getHeartRateFromGoogleFit();
  }

@override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Health Tracker"),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _initServices,
          ),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _buildCard("Passi", "$_stepCount", Icons.directions_walk, Colors.green),
          _buildCard("Velocità", "${_speed?.toStringAsFixed(2) ?? "N/A"} m/s", Icons.speed, Colors.blue),
          _buildCard("Battito cardiaco", "${_heartRate ?? "N/A"} bpm", Icons.favorite, Colors.red),
          Text(_heartLog, style: const TextStyle(fontSize: 14, color: Colors.black87)),
          _buildCard(
            "Posizione",
            "${_latitude?.toStringAsFixed(4) ?? "N/A"}, ${_longitude?.toStringAsFixed(4) ?? "N/A"}",
            Icons.location_on,
            Colors.orange,
          ),
        ],
      ),
    );
  }

  Widget _buildCard(String title, String value, IconData icon, Color color) {
    return Card(
      elevation: 4,
      margin: const EdgeInsets.only(bottom: 16),
      child: ListTile(
        leading: Icon(icon, color: color, size: 40),
        title: Text(title, style: const TextStyle(fontWeight: FontWeight.bold)),
        subtitle: Text(value, style: const TextStyle(fontSize: 18)),
      ),
    );
  }
}